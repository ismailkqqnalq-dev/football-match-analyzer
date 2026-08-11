#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to verify field point tracker system works"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.field_point_tracker import FieldPointTracker, FieldPoint
import json
import cv2
import numpy as np
import tempfile

def test_system():
    """Test all components of the field point tracking system"""
    
    print("\n" + "="*60)
    print("FIELD POINT TRACKER - SISTEM TEST")
    print("="*60 + "\n")
    
    # Test 1: Tracker olustur
    print("[Test 1] FieldPointTracker olusturuldu...", end=" ")
    tracker = FieldPointTracker(use_line_detection=True)
    print("OK")
    
    # Test 2: Test frame
    print("[Test 2] Test frame olusturuldu...", end=" ")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    print("OK")
    
    # Test 3: Tespit
    print("[Test 3] Noktalar tespit ediliyor...", end=" ")
    points = tracker.detect_field_points_in_frame(frame, 0)
    print(f"OK ({len(points)} nokta)")
    
    # Test 4: Scene cut detection
    print("[Test 4] Sahne kesintisi kontrolu...", end=" ")
    is_cut = tracker.detect_scene_cut(frame)
    print(f"OK (kesinti: {is_cut})")
    
    # Test 5: Nokta cizme
    print("[Test 5] Noktalar frame'e ciziliyor...", end=" ")
    annotated = tracker.draw_field_points(frame.copy())
    print(f"OK (cikti: {annotated.shape})")
    
    # Test 6: JSON export
    print("[Test 6] JSON export test...", end=" ")
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, 'test.json')
        tracker.export_to_json(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert 'field_points' in data
        assert 'metadata' in data
    print("OK")
    
    # Test 7: Pickle export
    print("[Test 7] Pickle export test...", end=" ")
    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = os.path.join(tmpdir, 'test.pkl')
        tracker.export_to_pickle(pkl_path)
        assert os.path.exists(pkl_path)
    print("OK")
    
    # Test 8: Main.py integration
    print("[Test 8] main.py entegrasyon kontrol...", end=" ")
    with open('main.py', encoding='utf-8') as f:
        content = f.read()
        assert 'FieldPointTracker' in content
        assert 'detect_field_points_in_frame' in content
        assert 'detect_scene_cut' in content
        assert 'field_point_tracker' in content
    print("OK")
    
    print("\n" + "="*60)
    print("TÜMMM TESTLER BAŞARILI!")
    print("="*60 + "\n")
    
    print("SISTEM HAZIR:")
    print("  1. python main.py --input video.mp4 --output output.mp4")
    print("  2. python utils/detect_field_points.py --input video.mp4")
    print("  3. Kalibrasyon: python utils/calibrate_homography.py")
    print("\n")

if __name__ == '__main__':
    try:
        test_system()
    except Exception as e:
        print(f"\nHATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
