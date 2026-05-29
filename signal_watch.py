import time
import json
import ctypes
import os
from pathlib import Path
from datetime import datetime
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
import undetected_chromedriver as uc
import MetaTrader5 as mt5

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
CHART_URL = "https://portal.dhanayantracharts.com/"
TARGET_ENDPOINT = "/api/chart/entry-signals"
BROWSER_NAME = "chrome"

CHROME_PROFILE_DIR = Path.home() / "AppData" / "Local" / "signal_watcher_chrome_profile"
EDGE_PROFILE_DIR = Path.home() / "AppData" / "Local" / "signal_watcher_edge_profile"

# MT5 Asset & Risk Controls
TRADE_SYMBOL = "XAUUSD"
LOT_SIZE = 1.0       # 0.01 Micro Lot = 1 ounce of Gold
SLIPPAGE = 10            # Max allowed price deviation points for execution


def get_local_chrome_major_version():
    """Return the installed Chrome major version, or None if it cannot be detected."""
    candidate_roots = [
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMFILES(X86)"),
        os.environ.get("LOCALAPPDATA"),
    ]

    for root in candidate_roots:
        if not root:
            continue

        chrome_path = Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe"
        if not chrome_path.exists():
            continue

        try:
            version_size = ctypes.windll.version.GetFileVersionInfoSizeW(str(chrome_path), None)
            if not version_size:
                continue

            version_buffer = ctypes.create_string_buffer(version_size)
            if not ctypes.windll.version.GetFileVersionInfoW(str(chrome_path), 0, version_size, version_buffer):
                continue

            lptr = ctypes.c_void_p()
            lsize = ctypes.c_uint()
            if not ctypes.windll.version.VerQueryValueW(version_buffer, r"\\", ctypes.byref(lptr), ctypes.byref(lsize)):
                continue

            class VS_FIXEDFILEINFO(ctypes.Structure):
                _fields_ = [
                    ("dwSignature", ctypes.c_uint32),
                    ("dwStrucVersion", ctypes.c_uint32),
                    ("dwFileVersionMS", ctypes.c_uint32),
                    ("dwFileVersionLS", ctypes.c_uint32),
                    ("dwProductVersionMS", ctypes.c_uint32),
                    ("dwProductVersionLS", ctypes.c_uint32),
                    ("dwFileFlagsMask", ctypes.c_uint32),
                    ("dwFileFlags", ctypes.c_uint32),
                    ("dwFileOS", ctypes.c_uint32),
                    ("dwFileType", ctypes.c_uint32),
                    ("dwFileSubtype", ctypes.c_uint32),
                    ("dwFileDateMS", ctypes.c_uint32),
                    ("dwFileDateLS", ctypes.c_uint32),
                ]

            file_info = ctypes.cast(lptr.value, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
            major = file_info.dwFileVersionMS >> 16
            if major:
                return int(major)
        except Exception:
            continue

    return None


def initialize_mt5():
    """Initializes connection to the running MT5 terminal instance."""
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed! Error code: {mt5.last_error()}")
        return False
    
    # Check if the symbol is available in market watch
    selected = mt5.symbol_select(TRADE_SYMBOL, True)
    if not selected:
        print(f"❌ Symbol {TRADE_SYMBOL} not found or could not be selected in MT5.")
        mt5.shutdown()
        return False
        
    print(f"✅ Successfully linked to MT5 Terminal. Tracking asset: {TRADE_SYMBOL}")
    return True


def close_existing_positions(symbol=TRADE_SYMBOL):
    """
    Scans for open positions on the symbol and explicitly closes them
    by ticket ID. Essential for Hedging accounts.
    """
    positions = mt5.positions_get(symbol=symbol)
    
    if positions is None:
        print(f"⚠️ Failed to retrieve MT5 positions for {symbol}. Error: {mt5.last_error()}")
        return False
        
    if len(positions) == 0:
        return True # Nothing to close, clear to proceed
        
    print(f"🧹 Found {len(positions)} open position(s) for {symbol}. Liquidating...")
    
    all_closed = True
    for pos in positions:
        tick = mt5.symbol_info_tick(symbol)
        
        # Determine the opposite action needed to close the ticket
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,           # <--- CRITICAL: explicitly targets the open trade
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "price": price,
            "slippage": SLIPPAGE,
            "comment": "Dhanayantri Reverse Close",
            "type_filling": mt5.ORDER_FILLING_FOK, 
        }

        result = mt5.order_send(request)
        
        # Standard Fallbacks for filling mode rejections
        if result and (result.retcode == 10030 or "filling" in str(result.comment).lower()):
            request["type_filling"] = mt5.ORDER_FILLING_RETURN
            result = mt5.order_send(request)
            
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Failed to close ticket #{pos.ticket}. Retcode: {getattr(result, 'retcode', 'NONE')}")
            all_closed = False
        else:
            print(f"✅ Successfully closed ticket #{pos.ticket} at {result.price}")
            
    return all_closed


def execute_mt5_order(order_type, lot_size, symbol=TRADE_SYMBOL):
    """
    Sends raw market orders directly to the live MT5 Execution Engine.
    order_type: mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL
    """
    # Fetch current market price tick
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"❌ Failed to get current price tick for {symbol}. Order aborted.")
        return False

    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    order_side = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
    print(f"🧾 [MT5 ORDER READY] Sending {order_side} {symbol} | volume={lot_size} | price={price}")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot_size),
        "type": order_type,
        "price": float(price),
        "slippage": SLIPPAGE,
        "comment": "Dhanayantri Auto-Signal Router",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,  
    }

    # Transmit execution request to terminal pipeline
    result = mt5.order_send(request)
    
    # 1. FIX: NoneType Protection Guard
    if result is None:
        print("❌ MT5 critical communication failure! order_send returned None. Skipping execution safety frame.")
        return False
        
    # 2. FIX: Automated Filling Mode Fallback (Handles Retcode 10030 / Unsupported filling mode)
    if result.retcode == 10030 or "filling" in str(result.comment).lower():
        print("⚠️ Broker rejected ORDER_FILLING_IOC. Attempting fallback to ORDER_FILLING_FOK...")
        request["type_filling"] = mt5.ORDER_FILLING_FOK
        result = mt5.order_send(request)
        
        if result is None:
            print("❌ MT5 critical communication failure during FOK fallback!")
            return False
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("⚠️ ORDER_FILLING_FOK also failed. Attempting final fallback to ORDER_FILLING_RETURN...")
            request["type_filling"] = mt5.ORDER_FILLING_RETURN
            result = mt5.order_send(request)
            
            if result is None:
                print("❌ MT5 critical communication failure during RETURN fallback!")
                return False

    # 3. Final verification check for the transaction state
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ MT5 Order Execution Failed! Retcode: {result.retcode} | Error text: {result.comment}")
        return False

    print(f"💰 [MT5 FILL CONFIRMED] Ticket #{result.order} | Executed {symbol} at {result.price}")
    return True


def get_response_body(driver, request_id):
    """Fetch network response body through Developer Tools instance."""
    try:
        response = driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": request_id})
    except Exception:
        return None

    body = response.get("body")
    if body is None:
        return None

    if response.get("base64Encoded"):
        try:
            import base64
            body = base64.b64decode(body).decode("utf-8", errors="replace")
        except Exception:
            return None

    return body


def open_chart_url(driver, url):
    """Navigate the active Chrome tab to the portal and recover if the first load stays blank."""
    try:
        driver.set_page_load_timeout(60)
    except Exception:
        pass

    try:
        driver.get(url)
    except Exception as exc:
        print(f"⚠️ Primary navigation attempt failed: {exc}")

    time.sleep(2)

    try:
        current_url = (driver.current_url or "").strip().lower()
    except Exception:
        current_url = ""

    if current_url in {"about:blank", "chrome://newtab/", "chrome://newtab"}:
        print("ℹ️ Chrome opened a blank tab. Retrying portal navigation in the same window...")
        try:
            driver.execute_script("window.location.href = arguments[0];", url)
        except Exception as exc:
            print(f"⚠️ JavaScript navigation retry failed: {exc}")

        time.sleep(2)

        try:
            current_url = (driver.current_url or "").strip().lower()
        except Exception:
            current_url = ""

        if current_url in {"about:blank", "chrome://newtab/", "chrome://newtab"}:
            print("ℹ️ Still blank. Opening the portal in a new tab and switching to it...")
            try:
                driver.execute_script("window.open(arguments[0], '_blank');", url)
                driver.switch_to.window(driver.window_handles[-1])
            except Exception as exc:
                print(f"⚠️ Fallback tab-open failed: {exc}")


def create_browser_options(browser_name):
    browser_name = browser_name.lower().strip()

    if browser_name == "edge":
        options = webdriver.EdgeOptions()
        options.add_argument('--start-maximized')
        options.add_argument('--disable-gpu')
        options.add_argument(f'--user-data-dir={EDGE_PROFILE_DIR}')
        options.add_argument('--profile-directory=Default')
        options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
    else:
        options = uc.ChromeOptions()
        options.add_argument('--start-maximized')
        options.add_argument('--disable-gpu')
        options.add_argument(f'--user-data-dir={CHROME_PROFILE_DIR}')
        options.add_argument('--profile-directory=Default')
        options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    options.set_capability('acceptInsecureCerts', True)
    return options


def monitor_browser_network():
    # Attempt to stitch into MT5 local engine before loading browser profiles
    if not initialize_mt5():
        return

    print(f"🚀 Launching normal visible {BROWSER_NAME.capitalize()} browser window...")
    options = create_browser_options(BROWSER_NAME)
    
    try:
        if BROWSER_NAME.lower().strip() == "edge":
            driver = webdriver.Edge(options=options)
        else:
            chrome_version = get_local_chrome_major_version()
            if chrome_version is not None:
                print(f"ℹ️ Detected local Chrome version: {chrome_version}")
                driver = uc.Chrome(options=options, version_main=chrome_version)
            else:
                print("ℹ️ Could not detect local Chrome version. Using undetected_chromedriver default.")
                driver = uc.Chrome(options=options)
            
        driver.execute_cdp_cmd('Network.enable', {})
    except Exception as e:
        print(f"❌ Error launching {BROWSER_NAME.capitalize()} Driver: {e}")
        mt5.shutdown()
        return
    
    print(f"🌐 Loading target portal wire: {CHART_URL}")
    open_chart_url(driver, CHART_URL)
    
    print("\n" + "="*70)
    print("👉 ACTION REQUIRED:")
    print("1. Complete manual credential authorization inside the browser window.")
    print(f"2. Ensure the main workspace has your active {TRADE_SYMBOL} chart open.")
    print("\n⚙️ MT5 Bridge Core Live. Intercepting telemetry wire signals...")
    print("="*70 + "\n")

    last_signal_time = None
    processed_request_ids = set()
    current_position = None  # Expected tracked states: None, "long", "short"

    try:
        while True:
            for entry in driver.get_log('performance'):
                try:
                    message = json.loads(entry['message'])['message']
                except Exception:
                    continue

                if message.get('method') != 'Network.responseReceived':
                    continue

                params = message.get('params', {})
                response = params.get('response', {})
                request_id = params.get('requestId')
                response_url = response.get('url', '')

                if not request_id or request_id in processed_request_ids:
                    continue

                if TARGET_ENDPOINT not in response_url:
                    continue

                processed_request_ids.add(request_id)

                try:
                    response_body = get_response_body(driver, request_id)
                    if not response_body:
                        continue

                    data = json.loads(response_body)

                    if "signals" in data and len(data["signals"]) > 0:
                        latest_signal = data["signals"][-1]
                        current_time = latest_signal["time"]
                        current_side = latest_signal["side"].lower().strip()  # "long" or "short"
                        signal_label = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

                        # Step 1: Establish baseline configuration parameters to ignore expired flags
                        if last_signal_time is None:
                            last_signal_time = current_time
                            readable_baseline = datetime.fromtimestamp(last_signal_time).strftime('%Y-%m-%d %H:%M:%S')
                            current_position = current_side
                            print(f"✅ Synchronization Verified. Last chart arrow instance: {readable_baseline} [{current_side.upper()}]")
                            print(f"📋 Tracker state matched to existing position context: {current_position.upper()}")

                        # Step 2: Act on real-time live execution updates
                        elif current_time > last_signal_time:
                            previous_time = last_signal_time
                            previous_side = current_position

                            last_signal_time = current_time
                            converted_time = signal_label

                            print("\n" + "⚡" * 25)
                            print(f"🚨 FRESH DHANAYANTRI SIGNAL INTERCEPTED 🚨")
                            print(f"📅 Clock Signature: {converted_time}")
                            print(f"📊 Signal Variant:  {current_side.upper()}")
                            if previous_time is not None and previous_side is not None:
                                previous_label = datetime.fromtimestamp(previous_time).strftime('%Y-%m-%d %H:%M:%S')
                                print(f"🕘 Last Signal Before This One: {previous_label} [{previous_side.upper()}]")
                            print("⚡" * 25 + "\n")

                            # --- AUTOMATED STOP AND REVERSE MT5 LOGIC BLOCK ---
                            
                            if current_side == current_position:
                                print(f"ℹ️ Position already exists for {current_position.upper()}. Dropping duplicate event frame.")
                            
                            else:
                                print(f"🔄 Direction change detected! Reversing state from {str(current_position).upper()} to {current_side.upper()}.")
                                
                                # 1. Explicitly close ANY opposing open positions by ticket
                                print(f"🛑 [MT5 EXECUTION] Flattening older trade allocations...")
                                close_verified = close_existing_positions(symbol=TRADE_SYMBOL)

                                if not close_verified:
                                    print("⚠️ Close leg failed or partially failed. Skipping the new entry to avoid overlapping/locked positions.")
                                    continue
                                
                                # 2. Open fresh direction position block
                                entry_type = mt5.ORDER_TYPE_BUY if current_side == "long" else mt5.ORDER_TYPE_SELL
                                print(f"🎬 [MT5 EXECUTION] Spinning up fresh {current_side.upper()} order execution profile...")
                                execution_verified = execute_mt5_order(order_type=entry_type, lot_size=LOT_SIZE)
                                
                                if execution_verified:
                                    current_position = current_side
                                    print(f"🎯 State tracker updated. Current allocation focus: {current_position.upper()}")

                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
                except Exception as e:
                    print(f"⚠️ Internal parser exception encountered: {e}")
            
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n👋 Bridge connection terminated manually by operator workflow.")
    except WebDriverException as e:
        print(f"\n⚠️ Browser session dropped structural connection: {e}")
    finally:
        print("Safely decoupling environmental parameters...")
        try:
            driver.quit()
        except:
            pass
        print("Disconnecting MT5 interface terminal...")
        mt5.shutdown()


if __name__ == "__main__":
    monitor_browser_network()