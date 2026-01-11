# windows.spec
# Build (PowerShell):
#   Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
#   uv pip install pyinstaller
#   uv run pyinstaller --clean windows.spec
#
# Run:
#   .\dist\palinstrophy\palinstrophy.exe
#

block_cipher = None

a = Analysis(
    ["palinstrophy/turbo_main.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="palinstrophy",
    console=True,
    icon="palinstrophy/palinstrophy.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="palinstrophy",
)