#!/usr/bin/env python3
"""check_shots -- the EP0 v5 shot-list gate (brief §4.1). Runs on the shot JSON BEFORE any render.

Every rule here is one of the owner's v4 complaints turned into something a script can fail on:

  R1 ADJACENCY       two consecutive shots must differ in at least TWO of
                     {subject, shot size, angle, camera move}. v4's first three shots were three
                     close-ups of the same face -- "they might as well be one shot".
  R2 CAMERA          no shot is locked off unless it is explicitly flagged `locked_off_allowed`
                     AND the shot is <= 6 s. Every shot names a camera motion type from the vendor
                     list with an amplitude and a speed. v4 held the camera on 19 of 28.
  R3 EMITTER         every shot that lists an emitter must have `in_frame` true on all of them, and
                     the prompt's action clause must name the origin. v4's m3_08 had fire that did
                     not come out of the dragon's mouth.
  R4 MOTION CLAUSE   present, opens already in motion, names the visible evidence of speed.
  R5 DIRECTION       screen direction declared, and it uses the film's screen-left/right vocabulary.
  R6 SPEED WORDS     each shot declares exactly one speed register (fast|slow) and the film is not
                     all fast (>=30% of shots must be slow) -- calling everything fast flattens the
                     dynamic range (2026-08-18).
  R7 LENGTH GRID     frame counts sit on the model's 17k+5 grid and inside the brief's 6-10 s band
                     (141/158/175/192/209/226/243).
  R8 BEATS           a shot of 8 s or longer carries at least two internal timed beats, in order,
                     inside the shot's own duration.
  R9 REFS            the blade sheet is attached to nothing; a shot whose subject is her face alone
                     carries the hero sheet; a shot with the animal in frame carries the mount sheet.

Usage: tools/shot_rules.py output/trailer_test/v5/shots_v5.json [--json OUT]
Exit 1 on any violation.
"""
import argparse, json, sys
from pathlib import Path

CAMERA_TYPES = {
 'Zoom In', 'Zoom Out', 'Push In', 'Pull Out', 'Pan Left', 'Pan Right', 'Truck Left', 'Truck Right',
 'Tilt Up', 'Tilt Down', 'Pedestal Up', 'Pedestal Down', 'Arc Shot', 'Tracking Shot', 'Static Shot',
 'Shake Slightly', 'Shake Strongly', 'POV', 'Roll Clockwise', 'Roll Counterclockwise',
}
AMPS = {'small amplitude', 'large amplitude', 'medium amplitude'}
SPEEDS = {'slow speed', 'fast speed', 'normal speed'}
GRID = {141, 158, 175, 192, 209, 226, 243}
EVIDENCE = ('hair', 'cloak', 'skirt', 'ember', 'ash', 'cloud', 'ground', 'smoke', 'wall', 'roof',
            'merlon', 'banner', 'robe', 'wing', 'streak', 'figures', 'fire')
ALREADY_MOVING = ('already', 'for the whole take', 'continuously', 'keeps', 'is at full')
BLADE = 'ref_vesta_blades'
HERO = 'ref_vestia_v3'
MOUNT = 'ref_vesta_rider_mount'
FACE_SUBJECTS = ('vesta_face', 'vesta_shoulder_and_cloud')
MOUNT_SUBJECTS = ('rider_mount', 'mount_pov', 'street_and_mount', 'city_wall_and_mount')


def check(doc):
    shots = list(doc['shots'].values())
    bad = []

    def fail(sid, rule, msg):
        bad.append('%-6s %s  %s' % (sid, rule, msg))

    # ---- R1 adjacency
    for a, b in zip(shots, shots[1:]):
        diff = [k for k in ('subject', 'shot_size', 'angle', 'camera_type') if a[k] != b[k]]
        if len(diff) < 2:
            fail(b['id'], 'R1', 'differs from %s in only %d of {subject,size,angle,camera}: %s'
                 % (a['id'], len(diff), diff or 'nothing'))

    for s in shots:
        sid = s['id']
        # ---- R2 camera
        if s['camera_type'] == 'Static Shot' or s.get('locked_off'):
            if not s.get('locked_off_allowed'):
                fail(sid, 'R2', 'is locked off and is not flagged locked_off_allowed')
            elif s['dur'] > 6.0:
                fail(sid, 'R2', 'is locked off at %.2f s -- allowed only at <= 6 s' % s['dur'])
        if s['camera_type'] not in CAMERA_TYPES:
            fail(sid, 'R2', 'camera_type %r is not in the vendor list' % s['camera_type'])
        if s['camera_amp'] not in AMPS:
            fail(sid, 'R2', 'camera_amp %r is not a vendor amplitude' % s['camera_amp'])
        if s['camera_speed'] not in SPEEDS:
            fail(sid, 'R2', 'camera_speed %r is not a vendor speed' % s['camera_speed'])
        cc = s['camera_clause']
        if s['camera_amp'] not in cc or s['camera_speed'] not in cc:
            fail(sid, 'R2', 'the camera clause does not carry both the amplitude and the speed')
        if not any(w in cc.lower() for w in s['camera_type'].lower().split()):
            fail(sid, 'R2', 'the camera clause does not name the motion type %r' % s['camera_type'])

        # ---- R3 emitter
        for e in s.get('emitters', []):
            if not e.get('in_frame'):
                fail(sid, 'R3', 'emitter %r is not in frame' % e['what'])
            if 'in frame' not in e['origin'] and 'in frame' not in s['action_clause']:
                fail(sid, 'R3', 'neither the emitter origin nor the action clause says "in frame"')
        if not s.get('emitters'):
            fail(sid, 'R3', 'no emitter declared -- every v5 shot has something that emits')

        # ---- R4 motion clause
        mc = s.get('motion_clause', '')
        if len(mc.split()) < 25:
            fail(sid, 'R4', 'motion clause is %d words -- too thin to carry the action' % len(mc.split()))
        if not any(w in mc.lower() for w in ALREADY_MOVING):
            fail(sid, 'R4', 'motion clause does not open already in motion')
        if not any(w in mc.lower() for w in EVIDENCE):
            fail(sid, 'R4', 'motion clause names no visible evidence of speed')
        if 'slowly begins' in mc.lower() or 'starts to' in mc.lower():
            fail(sid, 'R4', 'motion clause ramps up instead of opening at speed')

        # ---- R5 screen direction
        sd = s.get('screen_direction', '')
        if not sd or not any(w in sd.lower() for w in ('screen-left', 'screen-right', 'toward the lens',
                                                       'away from the lens', 'into the depth')):
            fail(sid, 'R5', 'screen direction is not declared in screen-left/right terms')

        # ---- R6 speed register
        if s.get('speed') not in ('fast', 'slow'):
            fail(sid, 'R6', 'speed register %r is not fast|slow' % s.get('speed'))
        else:
            want = 'high speed' if s['speed'] == 'fast' else 'slow speed'
            blob = (mc + ' ' + s['camera_clause']).lower()
            if want not in blob and s['speed'] + ' speed' not in blob:
                fail(sid, 'R6', 'declared %s but neither "%s" nor "%s speed" appears'
                     % (s['speed'], want, s['speed']))

        # ---- R7 length
        if s['frames'] not in GRID:
            fail(sid, 'R7', '%d frames is not on the 17k+5 grid inside the 6-10 s band' % s['frames'])

        # ---- R8 beats
        beats = s.get('beats', [])
        if s['dur'] >= 8.0 and len(beats) < 2:
            fail(sid, 'R8', 'is %.2f s with only %d internal beat(s)' % (s['dur'], len(beats)))
        ts = [b['t'] for b in beats]
        if ts != sorted(ts):
            fail(sid, 'R8', 'internal beats are not in time order')
        for b in beats:
            if not 0 < b['t'] < s['dur']:
                fail(sid, 'R8', 'beat at %.3f s falls outside the shot (0-%.3f s)' % (b['t'], s['dur']))

        # ---- R9 refs
        refs = ' '.join(s['refs_needed'])
        if BLADE in refs:
            fail(sid, 'R9', 'the blade sheet is attached -- it must be attached to nothing in v5')
        if s['subject'] in FACE_SUBJECTS and HERO not in refs:
            fail(sid, 'R9', 'is a her-only shot and must carry the hero sheet')
        if any(s['subject'].startswith(m) for m in MOUNT_SUBJECTS) and MOUNT not in refs:
            fail(sid, 'R9', 'has the animal in frame and must carry the mount sheet')
        if HERO in refs and MOUNT in refs:
            fail(sid, 'R9', 'carries both the hero and the mount sheet -- that asserts her twice')

    # ---- R6 film-level
    fast = sum(1 for s in shots if s.get('speed') == 'fast')
    if fast > 0.7 * len(shots):
        bad.append('FILM   R6  %d of %d shots are fast -- calling everything fast flattens the range'
                   % (fast, len(shots)))
    if len(shots) < 16 or len(shots) > 18:
        bad.append('FILM   R7  %d shots -- the brief asks for 16-18' % len(shots))
    return bad, dict(shots=len(shots), fast=fast, slow=len(shots) - fast,
                     total_s=doc['_total_s'], frames=doc['_total_frames'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shots')
    ap.add_argument('--json')
    a = ap.parse_args()
    doc = json.loads(Path(a.shots).read_text())
    bad, summ = check(doc)
    print('check_shots on %s' % a.shots)
    print('  %(shots)d shots, %(frames)d frames, %(total_s).3f s; %(fast)d fast / %(slow)d slow' % summ)
    if bad:
        print('\n%d VIOLATIONS' % len(bad))
        for b in bad:
            print('  * ' + b)
    else:
        print('\n  ALL RULES PASS')
    if a.json:
        Path(a.json).write_text(json.dumps({'summary': summ, 'violations': bad}, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
