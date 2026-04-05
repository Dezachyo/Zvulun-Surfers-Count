# ---------------- Run Settings ----------------
SAVE_DATA = True
TRACK_EXPIRED_FRAMES = 10
SAMPLING_DURATION_SEC = 20
SAMPLING_INTERVAL_MIN = 5

STREAM_URL = "https://vod.wavehub.co.il/live/_definst_/Zvulun_SD.stream/playlist.m3u8"
#STREAM_URL = "https://vod.wavehub.co.il/live/_definst_/Zvulun_1080p.stream/playlist.m3u8"

WG_URL = 'https://www.windguru.cz/2354'

TZ = "Asia/Jerusalem"


# Organize by Location ID
WEBCAMS = {
    "zvulun": {
        "name": "Zvulun Beach, Herzliya",
        "windguru_url": "https://www.windguru.cz/2354",
        "stream_url_sd": "https://vod.wavehub.co.il/live/_definst_/Zvulun_SD.stream/playlist.m3u8",
        "stream_url_hd": "https://vod.wavehub.co.il/live/_definst_/Zvulun_1080p.stream/playlist.m3u8",
        "lat": 32.1624,
        "lon": 34.8016
    },
    "marina": {
        "name": "Herzliya Marina",
        "windguru_url": "https://www.windguru.cz/xxxx",
        "stream_url_sd": "https://example.com/marina_low.m3u8",
        "stream_url_hd": "https://example.com/marina_high.m3u8",
        "lat": 32.1580,
        "lon": 34.7970
    }
}