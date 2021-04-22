#!/bin/bash
# Builds a macOS app in a DMG container using PyInstaller and an app launcher.
# usage (append e.g. "App" to name to avoid naming conflicts with the library):
#
#     osx_build_app.sh AppNameApp [AppVersion]
#
# Notes:
# - AppVersion is optional (used for name of DMG container)
# - This script must be called from the root directory of the repository
# - The file ./travis/AppNameApp.py [sic] must be present (relative
#   to root of the repository)

set -e

if [ -z $1 ]; then
    echo "Please specify package name as command line argument!"
    exit 1
fi
NAME=$1

if [ -z $2 ]; then
    NAMEVERSION=${1}
else
    NAMEVERSION=${1}_${2}
fi

SPEC="./macos_${NAME}.spec"
APP="./dist/${NAME}.app"
DMG="./dist/${NAMEVERSION}.dmg"
PKG="./dist/${NAME}.pkg"
TMP="./dist/pack.temp.dmg"

# cleanup from previous builds
rm -rf ./build
rm -rf ./dist

pip install -r macos_build_requirements.txt

pyinstaller -y --log-level=WARN ${SPEC}

# Test the binary by executing it with --version argument
echo ""
echo "...Testing the app (this should print the version)."
./dist/${NAME}.app/Contents/MacOS/${NAME}.bin --version
echo ""

# Create PKG (pkgbuild is for deployments in app stores)
# https://www.manpagez.com/man/1/productbuild/
#productbuild --install-location /Applications/ --component ${APP} ${PKG}
# https://www.manpagez.com/man/1/pkgbuild/
pkgbuild --install-location /Applications/ --component ${APP} ${PKG}

# Create DMG
# add link to Applications
mkdir ./dist/ui-release
cd ./dist/ui-release
ln -s /Applications
cd -
mv ${APP} ./dist/ui-release/

# create temporary DMG
hdiutil create -srcfolder ./dist/ui-release/ -volname "${NAMEVERSION}" -fs HFS+ \
        -fsargs "-c c=64,a=16,e=16" -format UDRW "${TMP}"

# optional: edit the DMG
# https://stackoverflow.com/questions/96882/how-do-i-create-a-nice-looking-dmg-for-mac-os-x-using-command-line-tools

# create crompressed DMG
hdiutil convert "${TMP}" -format UDZO -imagekey zlib-level=9 -o "${DMG}"

# remove temporary DMG
rm $TMP

