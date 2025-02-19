import sys  # English: Import sys for platform checking / 日本語: プラットフォーム判定のためsysをインポート
try:
    if sys.platform == "win32":
        raise ImportError
    import resource  # English: Import resource on non-Windows platforms / 日本語: Windows以外はresourceをインポート
except ImportError:
    import psutil  # English: Fallback to psutil on Windows / 日本語: Windowsの場合はpsutilを使用 