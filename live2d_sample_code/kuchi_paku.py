import asyncio
import random
import pyvts

plugin_info = {
    "plugin_name": "SkynaController",
    "developer": "you",
    "authentication_token_path": "./token.txt"
}

# 音声再生中かどうかのフラグ(実際のTTS再生処理と連動させる)
is_speaking = False

async def set_mouth(vts, value):
    await vts.request(vts.vts_request.requestSetParameterValue(
        parameter="MouthOpen", value=value, face_found=True
    ))

async def mouth_flap_loop(vts):
    """喋っている間だけ口をパクパクさせる"""
    while True:
        if is_speaking:
            await set_mouth(vts, 1.0)  # 口を開ける
            await asyncio.sleep(0.15)
            await set_mouth(vts, 0.0)  # 口を閉じる
            await asyncio.sleep(0.15)
        else:
            await set_mouth(vts, 0.0)  # 喋ってないときは閉じたまま
            await asyncio.sleep(0.1)

async def blink_once(vts, close_time=0.08, open_time=0.08):
    await vts.request(vts.vts_request.requestSetParameterValue(
        parameter="EyeOpenLeft", value=0, face_found=True
    ))
    await vts.request(vts.vts_request.requestSetParameterValue(
        parameter="EyeOpenRight", value=0, face_found=True
    ))
    await asyncio.sleep(close_time)

    await vts.request(vts.vts_request.requestSetParameterValue(
        parameter="EyeOpenLeft", value=1, face_found=True
    ))
    await vts.request(vts.vts_request.requestSetParameterValue(
        parameter="EyeOpenRight", value=1, face_found=True
    ))
    await asyncio.sleep(open_time)

async def auto_blink_loop(vts):
    while True:
        await asyncio.sleep(random.uniform(2.0, 5.0))
        await blink_once(vts)

# --- 動作確認用：ダミーで喋らせてみるテスト ---
async def fake_speech_test():
    """実際のTTS再生の代わりに、5秒間喋っているフリをするテスト関数"""
    global is_speaking
    await asyncio.sleep(3)  # 3秒待ってから喋り始める
    print("喋り始め")
    is_speaking = True
    await asyncio.sleep(5)  # 5秒間喋る
    is_speaking = False
    print("喋り終わり")

async def main():
    vts = pyvts.vts(plugin_info=plugin_info)
    await vts.connect()
    await vts.request_authenticate_token()
    await vts.request_authenticate()

    print("接続成功。まばたき+口パクループを開始します(Ctrl+Cで停止)")

    # まばたき・口パク・テスト用の喋りを同時に走らせる
    await asyncio.gather(
        auto_blink_loop(vts),
        mouth_flap_loop(vts),
        fake_speech_test()
    )

if __name__ == "__main__":
    asyncio.run(main())