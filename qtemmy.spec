# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
hiddenimports += collect_submodules('nvidia-cuda-cupti-cu12')
hiddenimports += collect_submodules('nvidia-cuda-nvrtc-cu12')
hiddenimports += collect_submodules('nvidia-cuda-runtime-cu12')
hiddenimports += collect_submodules('nvidia-cufft-cu12')
hiddenimports += collect_submodules('nvidia-cufile-cu12')
hiddenimports += collect_submodules('nvidia-curand-cu12')
hiddenimports += collect_submodules('nvidia-cusolver-cu12')
hiddenimports += collect_submodules('nvidia-cusparse-cu12')
hiddenimports += collect_submodules('nvidia-cusparselt-cu12')
hiddenimports += collect_submodules('nvidia-nccl-cu12')
hiddenimports += collect_submodules('nvidia-nvjitlink-cu12')
hiddenimports += collect_submodules('nvidia-nvshmem-cu12')
hiddenimports += collect_submodules('nvidia-nvtx-cu12')


a = Analysis(
    ['/home/lainlives/stemmygit/qtemmy.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'tkinter', 'zoneinfo', 'blip2to3', 'Gtk', 'PySide2', 'PySide6', 'PySide5', 'PySide3', 'PySide4', 'setuptools_scm', 'jinja2', 'moviepy', 'yt-dlp'],
    noarchive=True,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [('v', None, 'OPTION')],
    exclude_binaries=False,
    name='qtemmy',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='qtemmy',
)
