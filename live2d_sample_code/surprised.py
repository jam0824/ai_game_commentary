import asyncio
import random
import pyvts

plugin_info = {
    "plugin_name": "SkynaController",
    "developer": "you",
    "authentication_token_path": "./token.txt"
}

is_speaking = False
vts_lock = asyncio.Lock()  # リクエストの順番待ち用ロック

async def safe_request(vts, msg):
    """複数のループから同時にリクエストが飛んでも衝突しないようにする"""
    async with vts_lock:
        return await vts.request(msg)

async def set_mouth(vts, value):
    await safe_request(vts, vts.vts_request.requestSetParameterValue(
        parameter="MouthOpen", value=value, face_found=True
    ))

async def mouth_flap_loop(vts):
    while True:
        if is_speaking:
            await set_mouth(vts, 1.0)
            await asyncio.sleep(0.15)
            await set_mouth(vts, 0.0)
            await asyncio.sleep(0.15)
        else:
            await asyncio.sleep(0.1)

async def blink_once(vts, close_time=0.08, open_time=0.08):
    await safe_request(vts, vts.vts_request.requestSetParameterValue(
        parameter="EyeOpenLeft", value=0, face_found=True
    ))
    await safe_request(vts, vts.vts_request.requestSetParameterValue(
        parameter="EyeOpenRight", value=0, face_found=True
    ))
    await asyncio.sleep(close_time)
    await safe_request(vts, vts.vts_request.requestSetParameterValue(
        parameter="EyeOpenLeft", value=1, face_found=True
    ))
    await safe_request(vts, vts.vts_request.requestSetParameterValue(
        parameter="EyeOpenRight", value=1, face_found=True
    ))
    await asyncio.sleep(open_time)

async def auto_blink_loop(vts):
    while True:
        await asyncio.sleep(random.uniform(2.0, 5.0))
        await blink_once(vts)

async def trigger_surprised(vts, hold_time=1.5):
    response = await safe_request(vts, vts.vts_request.requestHotKeyList())
    hotkeys = response["data"]["availableHotkeys"]
    target = next((h for h in hotkeys if h["name"] == "驚き"), None)

    if target is None:
        print("「驚き」ホットキーが見つかりません")
        return

    await safe_request(vts, vts.vts_request.requestTriggerHotKey(target["hotkeyID"]))
    print("驚き顔ON")

    await asyncio.sleep(hold_time)

    await safe_request(vts, vts.vts_request.requestTriggerHotKey(target["hotkeyID"]))
    print("驚き顔OFF")

async def demo_test(vts):
    global is_speaking
    await asyncio.sleep(3)
    is_speaking = True
    print("喋り始め")

    await asyncio.sleep(1)
    await trigger_surprised(vts, hold_time=2.0)

    await asyncio.sleep(3)
    is_speaking = False
    print("喋り終わり")

async def main():
    vts = pyvts.vts(plugin_info=plugin_info)
    await vts.connect()
    await vts.request_authenticate_token()
    await vts.request_authenticate()

    print("接続成功。まばたき+口パク+驚きテストを開始します(Ctrl+Cで停止)")

    await asyncio.gather(
        auto_blink_loop(vts),
        mouth_flap_loop(vts),
        demo_test(vts)
    )

if __name__ == "__main__":
    asyncio.run(main())