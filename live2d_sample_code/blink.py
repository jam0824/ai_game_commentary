import asyncio
import random
import pyvts

# プラグイン情報(名前は自由に決めてOK)
plugin_info = {
    "plugin_name": "SkynaController",
    "developer": "you",
    "authentication_token_path": "./token.txt"
}

async def blink_once(vts, close_time=0.08, open_time=0.08):
    """まばたきを1回行う"""
    # 目を閉じる
    await vts.request(vts.vts_request.requestSetParameterValue(
        parameter="EyeOpenLeft", value=0, face_found=True
    ))
    await vts.request(vts.vts_request.requestSetParameterValue(
        parameter="EyeOpenRight", value=0, face_found=True
    ))
    await asyncio.sleep(close_time)

    # 目を開ける
    await vts.request(vts.vts_request.requestSetParameterValue(
        parameter="EyeOpenLeft", value=1, face_found=True
    ))
    await vts.request(vts.vts_request.requestSetParameterValue(
        parameter="EyeOpenRight", value=1, face_found=True
    ))
    await asyncio.sleep(open_time)

async def auto_blink_loop(vts):
    """ランダムな間隔でまばたきを繰り返す"""
    while True:
        await asyncio.sleep(random.uniform(2.0, 5.0))  # 2〜5秒ごと
        await blink_once(vts)

async def main():
    vts = pyvts.vts(plugin_info=plugin_info)
    await vts.connect()

    # 初回のみ、VTube Studio側で「プラグインを許可する」ポップアップが出ます
    await vts.request_authenticate_token()
    await vts.request_authenticate()

    print("接続成功。まばたきループを開始します(Ctrl+Cで停止)")
    await auto_blink_loop(vts)

if __name__ == "__main__":
    asyncio.run(main())