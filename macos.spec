# macos.spec
# Build:
#   rm -rf build dist
#   uv pip install pyinstaller
#   uv run pyinstaller macos.spec
#   ./dist/palinstrophy.app/Contents/MacOS/palinstrophy
#   open -n ./dist/palinstrophy.app
#

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


mlx_binaries = collect_dynamic_libs("mlx")
mlx_datas = collect_data_files("mlx", includes=["lib/mlx.metallib"])

a = Analysis(
    ["palinstrophy/turbo_main.py"],
    pathex=["."],
    binaries=mlx_binaries,
    datas=mlx_datas,
    hiddenimports=["mlx._reprlib_fix"],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="palinstrophy",
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
    version="0.1.6",
    icon="palinstrophy/palinstrophy.icns",
    bundle_identifier="se.mannetroll.palinstrophy",
)
