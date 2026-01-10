# macos.spec
# Build:
#   rm -rf build dist
#   uv pip install pyinstaller
#   uv run pyinstaller macos.spec
#   ./dist/cupyturbo.app/Contents/MacOS/palinstrophy
#   open -n ./dist/palinstrophy.app
#

a = Analysis(
    ["palinstrophy/turbo_main.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="scipyturbo",
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="palinstrophy",
)

app = BUNDLE(
    coll,
    name="palinstrophy.app",
    icon="palinstrophy/palinstrophy.icns",
    bundle_identifier="se.mannetroll.palinstrophy",
)