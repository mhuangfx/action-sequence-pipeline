import subprocess, json, urllib.request
base = 'v2v_test/restyled/'
vids = ['chain30_v1.mp4', 'chain30_v3.mp4', 'duel_nox_3ref/video.mp4',
        'vertical_native/video.mp4', 'bullettime_v1/video.mp4', 'own_motion_test/video.mp4']
for v in vids:
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                          '-show_entries', 'stream=width,height',
                          '-show_entries', 'format=duration', '-of', 'json', base + v],
                         capture_output=True, text=True)
    d = json.loads(out.stdout)
    print(v, d['streams'][0]['width'], d['streams'][0]['height'], d['format']['duration'])
tags = json.load(urllib.request.urlopen('http://localhost:11434/api/tags'))
print([m['name'] for m in tags['models']])
