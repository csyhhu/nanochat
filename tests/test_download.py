#!/usr/bin/env python3
"""
Test script to verify the word list download works.
Tests both direct URL and ghproxy mirror.
"""

import sys
import os
import urllib.request
import tempfile
import time

# Set HF_ENDPOINT to use domestic mirror for faster download
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Add project root to path for importing common.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
MIRROR_URL = "https://ghproxy.com/" + WORD_LIST_URL

def test_download(url, label, timeout=30):
    """Test downloading from a URL."""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            content_length = response.headers.get('Content-Length', 'unknown')
            # Read first 1KB to verify it's text
            chunk = response.read(1024)
            content_size = response.read()  # Read rest
            total_size = len(chunk) + len(content_size)
            
            # Verify it looks like a word list
            first_line = chunk.split(b'\n')[0].decode('utf-8', errors='replace')
        
        elapsed = time.time() - start_time
        print(f"  Status: HTTP {status}")
        print(f"  Content-Length header: {content_length}")
        print(f"  Downloaded: {total_size:,} bytes")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  First word: {first_line}")
        print(f"  Result: [OK] Download successful!")
        return True
        
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start_time
        print(f"  HTTP Error: {e.code} {e.reason}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Result: [FAIL] HTTP {e.code}")
        return False
        
    except urllib.error.URLError as e:
        elapsed = time.time() - start_time
        print(f"  URL Error: {e.reason}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Result: [FAIL] {e.reason}")
        return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  Error: {type(e).__name__}: {e}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Result: [FAIL] {e}")
        return False

def test_common_module():
    """Test download via nanochat.common.download_file_with_lock."""
    print(f"\n{'='*60}")
    print(f"Testing: nanochat.common.download_file_with_lock")
    print(f"{'='*60}")
    
    try:
        from nanochat.common import download_file_with_lock, get_base_dir
        
        # Show current HF_ENDPOINT and cache dir
        hf_endpoint = os.environ.get("HF_ENDPOINT", "(not set)")
        print(f"  HF_ENDPOINT: {hf_endpoint}")
        print(f"  Cache dir: {get_base_dir()}")
        
        filename = "words_alpha.txt"
        file_path = download_file_with_lock(WORD_LIST_URL, filename, max_retries=1, timeout=30)
        
        # Verify file
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            print(f"  File: {file_path}")
            print(f"  Size: {size:,} bytes")
            print(f"  First word: {first_line}")
            print(f"  Result: [OK] Download via common module successful!")
            return True
        else:
            print(f"  Result: [FAIL] File not found after download")
            return False
            
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        print(f"  Result: [FAIL] {e}")
        return False

def test_spellingbee_init():
    """Test SpellingBee task initialization (which triggers download)."""
    print(f"\n{'='*60}")
    print(f"Testing: SpellingBee task initialization")
    print(f"{'='*60}")
    
    try:
        from tasks.spellingbee import SpellingBee
        task = SpellingBee(size=10, split="train")
        print(f"  Loaded {len(task.words):,} words")
        print(f"  First 5 words: {task.words[:5]}")
        print(f"  Result: [OK] SpellingBee initialized successfully!")
        return True
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print(f"  Result: [FAIL] {e}")
        return False

def main():
    print("=" * 60)
    print("  Word List Download Test")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Working dir: {os.getcwd()}")
    
    results = {}
    
    # Test 1: Direct download
    results['direct'] = test_download(WORD_LIST_URL, "Direct (raw.githubusercontent.com)")
    
    # Test 2: Mirror download
    results['mirror'] = test_download(MIRROR_URL, "Mirror (ghproxy.com)")
    
    # Test 3: Common module
    results['common'] = test_common_module()
    
    # Test 4: SpellingBee init
    results['spellingbee'] = test_spellingbee_init()
    
    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for test_name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {test_name:20s}: {status}")
    
    all_pass = all(results.values())
    if all_pass:
        print(f"\nAll tests passed!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\nSome tests failed: {failed}")
        print("\nTroubleshooting tips:")
        print("  1. If direct fails but mirror works: set HF_ENDPOINT to use hf-mirror")
        print("     PowerShell: $env:HF_ENDPOINT = 'https://hf-mirror.com'")
        print("  2. If both fail: check your network connection / proxy settings")
        print("  3. Manual download: download words_alpha.txt from:")
        print(f"     {WORD_LIST_URL}")
        print(f"     Save to: %USERPROFILE%\\.cache\\nanochat\\words_alpha.txt")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
