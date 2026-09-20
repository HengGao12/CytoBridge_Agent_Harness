import base64
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Thread
import webbrowser
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger("cytobridge_agent.gemini_oauth")

# Google Code Assist / Gemini CLI Constants
REDIRECT_URI = "http://localhost:8085/oauth2callback"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

TIER_FREE = "free-tier"
TIER_LEGACY = "legacy-tier"
TIER_STANDARD = "standard-tier"

# Credentials file storage
CREDENTIALS_FILE = Path.home() / ".cellcompass" / "gemini_credentials.json"
_CREDENTIALS_LOCK = threading.Lock()


def _generate_pkce() -> Tuple[str, str]:
    """Generate PKCE verifier and challenge."""
    verifier = os.urandom(32).hex()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode('utf-8')).digest()).decode('utf-8').rstrip('=')
    return verifier, challenge


def extract_gemini_cli_credentials() -> Optional[Tuple[str, Optional[str]]]:
    """Extracts OAuth credentials from the installed Gemini CLI's bundled oauth2.js."""
    # 1. Environment variables first
    client_id = os.getenv("OPENCLAW_GEMINI_OAUTH_CLIENT_ID") or os.getenv("GEMINI_CLI_OAUTH_CLIENT_ID")
    client_secret = os.getenv("OPENCLAW_GEMINI_OAUTH_CLIENT_SECRET") or os.getenv(
        "GEMINI_CLI_OAUTH_CLIENT_SECRET"
    )
    if client_id:
        return client_id, client_secret

    # 2. Try to find local gemini binary
    gemini_path = shutil.which("gemini")
    if not gemini_path:
        return None

    try:
        resolved_path = Path(gemini_path).resolve()
        gemini_cli_dir = resolved_path.parent.parent

        search_paths = [
            gemini_cli_dir / "node_modules" / "@google" / "gemini-cli-core" / "dist" / "src" / "code_assist" / "oauth2.js",
            gemini_cli_dir / "node_modules" / "@google" / "gemini-cli-core" / "dist" / "code_assist" / "oauth2.js",
        ]

        content = None
        for p in search_paths:
            if p.exists():
                content = p.read_text(encoding='utf-8')
                break

        if not content:
            # Fallback deep search
            for root, _, files in os.walk(gemini_cli_dir):
                if "oauth2.js" in files and "code_assist" in root:
                    content = Path(root, "oauth2.js").read_text(encoding='utf-8')
                    break

        if not content:
            return None

        id_match = re.search(r'(\d+-[a-z0-9]+\.apps\.googleusercontent\.com)', content)
        secret_match = re.search(r'(GOCSPX-[A-Za-z0-9_-]+)', content)

        if id_match and secret_match:
            return id_match.group(1), secret_match.group(1)

    except Exception as e:
        logger.debug(f"Failed to extract credentials from binary: {e}")

    return None


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Callback server handler to capture auth code."""
    def log_message(self, format, *args):
        pass  # suppress standard HTTP logging

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/oauth2callback":
            query = urllib.parse.parse_qs(parsed.query)
            
            error = query.get('error', [None])[0]
            if error:
                self.send_response(400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Authentication failed: {error}".encode('utf-8'))
                self.server.oauth_error = error
                return

            code = query.get('code', [None])[0]
            state = query.get('state', [None])[0]

            if not code or not state:
                self.send_response(400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Missing code or state")
                self.server.oauth_error = "Missing code or state"
                return

            if state != self.server.expected_state:
                self.send_response(400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Invalid state")
                self.server.oauth_error = "OAuth state mismatch"
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<!doctype html><html><head><meta charset='utf-8'/></head>"
                b"<body><h2>Gemini CLI OAuth complete</h2>"
                b"<p>You can close this window and return to CellCompass.</p></body></html>"
            )
            self.server.oauth_code = code
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not found")


def wait_for_local_callback(expected_state: str, timeout_seconds: int = 300) -> str:
    """Start local server and wait for the OAuth callback."""
    server = HTTPServer(('localhost', 8085), OAuthCallbackHandler)
    server.expected_state = expected_state
    server.oauth_code = None
    server.oauth_error = None

    def serve_forever(httpd):
        httpd.serve_forever()

    thread = Thread(target=serve_forever, args=(server,))
    thread.daemon = True
    thread.start()

    logger.info(f"Waiting for OAuth callback on {REDIRECT_URI}...")
    
    start_time = time.time()
    try:
        while time.time() - start_time < timeout_seconds:
            if server.oauth_code:
                return server.oauth_code
            if server.oauth_error:
                raise Exception(f"OAuth error: {server.oauth_error}")
            time.sleep(0.5)
        raise Exception("OAuth callback timeout")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def exchange_code_for_tokens(
    code: str,
    verifier: str,
    client_id: str,
    client_secret: Optional[str],
) -> Dict[str, Any]:
    """Exchanges auth code for tokens."""
    payload = {
        "client_id": client_id,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
    }
    if client_secret:
        payload["client_secret"] = client_secret
    data = urllib.parse.urlencode(payload).encode('utf-8')

    req = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    # Perform request
    with urllib.request.urlopen(req) as response:
        body = json.loads(response.read().decode('utf-8'))

    access_token = body.get("access_token")
    refresh_token = body.get("refresh_token")
    expires_in = body.get("expires_in", 3600)

    if not access_token or not refresh_token:
        raise Exception("Failed to retrieve tokens")

    # Initialize GCP project and Cloud Code configuration (Required for API to work)
    try:
        project_id = discover_and_onboard_project(access_token)
    except Exception as e:
        logger.warning(f"Could not automatically onboard GCP Project (you may need GOOGLE_CLOUD_PROJECT env var): {e}")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or "none"

    credentials = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": time.time() + expires_in - 300, # Pad 5 minutes
        "project_id": project_id,
        "client_id": client_id,
        "client_secret": client_secret
    }
    
    _save_credentials(credentials)
    return credentials


def refresh_access_token(credentials: Dict[str, Any]) -> str:
    """Uses the refresh token to get a new access token."""
    logger.info("Refreshing Gemini access token...")
    payload = {
        "client_id": credentials["client_id"],
        "refresh_token": credentials["refresh_token"],
        "grant_type": "refresh_token",
    }
    client_secret = credentials.get("client_secret")
    if client_secret:
        payload["client_secret"] = client_secret
    data = urllib.parse.urlencode(payload).encode('utf-8')

    req = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urllib.request.urlopen(req) as response:
        body = json.loads(response.read().decode('utf-8'))

    new_access_token = body.get("access_token")
    if not new_access_token:
        raise Exception("Failed to refresh token")

    credentials["access_token"] = new_access_token
    # Sometimes Google sends a new refresh token, sometimes they don't
    if "refresh_token" in body:
        credentials["refresh_token"] = body["refresh_token"]
    
    expires_in = body.get("expires_in", 3600)
    credentials["expires_at"] = time.time() + expires_in - 300
    
    _save_credentials(credentials)
    return new_access_token


def _save_credentials(credentials: Dict[str, Any]):
    CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _CREDENTIALS_LOCK:
        with NamedTemporaryFile("w", dir=str(CREDENTIALS_FILE.parent), delete=False) as tmp:
            json.dump(credentials, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, CREDENTIALS_FILE)
        os.chmod(CREDENTIALS_FILE, 0o600)


def _load_credentials() -> Dict[str, Any]:
    with _CREDENTIALS_LOCK:
        with open(CREDENTIALS_FILE, "r") as f:
            return json.load(f)


def get_valid_access_token() -> str:
    """Retrieves a valid access token, performing OAuth or refresh if needed."""
    if not CREDENTIALS_FILE.exists():
        logger.info("No Gemini credentials found, starting OAuth flow.")
        return login_gemini_cli_oauth()["access_token"]
    try:
        os.chmod(CREDENTIALS_FILE, 0o600)
    except Exception:
        pass

    try:
        credentials = _load_credentials()
            
        if time.time() >= credentials.get("expires_at", 0):
            return refresh_access_token(credentials)
            
        return credentials["access_token"]
        
    except Exception as e:
        logger.error(f"Failed to read/refresh credentials: {e}. Reigniting OAuth...")
        return login_gemini_cli_oauth()["access_token"]


def get_valid_auth_context() -> Dict[str, Any]:
    """Return valid auth context containing access token and project id."""
    token = get_valid_access_token()
    project_id = "none"
    if CREDENTIALS_FILE.exists():
        try:
            creds = _load_credentials()
            project_id = creds.get("project_id") or "none"
        except Exception:
            project_id = "none"
    return {
        "access_token": token,
        "project_id": project_id,
    }


def login_gemini_cli_oauth() -> Dict[str, Any]:
    """Perform the full OAuth loop."""
    creds = extract_gemini_cli_credentials()
    if not creds:
        raise Exception(
            "Gemini CLI credentials not found. Ensure @google/gemini-cli is "
            "installed (`npm i -g @google/gemini-cli`) or set GEMINI_CLI_OAUTH_CLIENT_ID "
            "(and optionally GEMINI_CLI_OAUTH_CLIENT_SECRET)."
        )

    client_id, client_secret = creds
    verifier, challenge = _generate_pkce()

    params = urllib.parse.urlencode({
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
        "access_type": "offline",
        "prompt": "consent",
    })

    auth_url = f"{AUTH_URL}?{params}"
    
    print("\n" + "=" * 60)
    print("Google Gemini CLI Authentication")
    print("=" * 60)
    print(f"\nIf your browser doesn't open automatically, navigate to:\n\n{auth_url}\n")
    print("Waiting for callback on localhost:8085...\n")
    
    try:
        webbrowser.open(auth_url)
    except:
        pass # Probably headless
        
    try:
        code = wait_for_local_callback(expected_state=verifier)
        return exchange_code_for_tokens(code, verifier, client_id, client_secret)
    except Exception as e:
        if "address already in use" in str(e).lower() or "EADDRINUSE" in str(e):
            if not sys.stdin.isatty():
                raise Exception(
                    "OAuth callback port 8085 is in use and non-interactive mode cannot prompt for manual redirect URL. "
                    "Free port 8085 or run with a TTY."
                )
            print("\nLocal callback server failed to start (port in use).")
            print(f"Please copy the full redirect URL you landed on and paste it below.")
            raw_url = input("Paste redirect URL: ").strip()
            
            try:
                parsed = urllib.parse.urlparse(raw_url)
                query = urllib.parse.parse_qs(parsed.query)
                manual_code = query['code'][0]
                manual_state = query['state'][0]
                if manual_state != verifier:
                    raise Exception("State mismatch")
                return exchange_code_for_tokens(manual_code, verifier, client_id, client_secret)
            except Exception as manual_e:
                raise Exception(f"Failed to parse manual URL: {manual_e}")
                
        raise e


# --- GCP Cloud Code Assist Provisioning (Copied from OpenClaw) ---
def discover_and_onboard_project(access_token: str) -> str:
    env_project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "gl-node/openclaw",
    }
    
    load_body = {
        "cloudaicompanionProject": env_project,
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
            "duetProject": env_project,
        },
    }
    
    req_load = urllib.request.Request(
        f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
        data=json.dumps(load_body).encode('utf-8'),
        headers=headers,
        method="POST"
    )
    
    data = {}
    try:
        with urllib.request.urlopen(req_load) as response:
            data = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8')
        # Check VPC SC violation
        if "SECURITY_POLICY_VIOLATED" in err_body:
            data = {"currentTier": {"id": TIER_STANDARD}}
        else:
            raise Exception(f"loadCodeAssist failed: {e.code} {e.reason} {err_body}")
            
    if "currentTier" in data:
        project = data.get("cloudaicompanionProject")
        if isinstance(project, str) and project:
            return project
        if isinstance(project, dict) and project.get("id"):
            return project["id"]
        if env_project:
            return env_project
        raise Exception("This account requires GOOGLE_CLOUD_PROJECT to be set.")

    # Onboarding
    tier = TIER_LEGACY
    allowed = data.get("allowedTiers", [])
    if allowed:
        for t in allowed:
            if t.get("isDefault"):
                tier = t.get("id", TIER_FREE)
                break
                
    if tier != TIER_FREE and not env_project:
        raise Exception("This account requires GOOGLE_CLOUD_PROJECT to be set.")
        
    onboard_body = {
        "tierId": tier,
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        },
    }
    if tier != TIER_FREE and env_project:
        onboard_body["cloudaicompanionProject"] = env_project
        onboard_body["metadata"]["duetProject"] = env_project
        
    req_onboard = urllib.request.Request(
        f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
        data=json.dumps(onboard_body).encode('utf-8'),
        headers=headers,
        method="POST"
    )
    
    with urllib.request.urlopen(req_onboard) as response:
        lro = json.loads(response.read().decode('utf-8'))
        
    if not lro.get("done") and lro.get("name"):
        operation_name = lro["name"]
        for _ in range(24):
            time.sleep(5)
            try:
                poll_req = urllib.request.Request(
                    f"{CODE_ASSIST_ENDPOINT}/v1internal/{operation_name}",
                    headers=headers
                )
                with urllib.request.urlopen(poll_req) as p_res:
                    lro = json.loads(p_res.read().decode('utf-8'))
                    if lro.get("done"):
                        break
            except:
                continue

    project_id = lro.get("response", {}).get("cloudaicompanionProject", {}).get("id")
    if project_id:
        return project_id
    if env_project:
        return env_project
        
    raise Exception("Could not discover or provision a Google Cloud project. Set GOOGLE_CLOUD_PROJECT.")
