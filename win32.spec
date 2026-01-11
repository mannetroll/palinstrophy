# win32.spec
# Build (PowerShell):
#   Remove-Item -Recurse -Force build,dist -ErrorAction SilentlyContinue
#   uv sync --extra cuda
#   uv pip install pyinstaller
#   uv run pyinstaller win32.spec --noconfirm --clean
#
# Run:
#   .\dist\palinstrophy\palinstrophy.exe

from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_dynamic_libs,
    collect_data_files,
)

# ---- CuPy (GPU extra) payload ----
cupy_hiddenimports = []
cupy_binaries = []
cupy_datas = []

cupy_hiddenimports += collect_submodules("cupy")
cupy_hiddenimports += collect_submodules("cupyx")
cupy_hiddenimports += collect_submodules("cupy_backends")

# ---- FIX: include fastrlock fully ----
cupy_hiddenimports += collect_submodules("fastrlock")
cupy_hiddenimports += ["fastrlock.rlock"]  # explicit, belt+braces
cupy_datas += collect_data_files("fastrlock", include_py_files=False)
cupy_binaries += collect_dynamic_libs("fastrlock")
# -------------------------------------

cupy_hiddenimports += ["cupy_backends.cuda._softlink"]

cupy_binaries += collect_dynamic_libs("cupy")
cupy_binaries += collect_dynamic_libs("cupy_backends")

cupy_datas += collect_data_files("cupy", include_py_files=False)
cupy_datas += collect_data_files("cupy_backends", include_py_files=False)

a = Analysis(
    ["palinstrophy/turbo_main.py"],
    pathex=["."],
    binaries=cupy_binaries,
    datas=cupy_datas,
    hiddenimports=cupy_hiddenimports,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="palinstrophy",
    console=True,   # keep True until runtime is clean; then set False
    icon="palinstrophy/palinstrophy.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="palinstrophy",
)