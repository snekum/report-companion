#!/usr/bin/env bash
# Downloads the pinned JRE and Synthea release into tools/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
JRE_URL="https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.13.1%2B2/OpenJDK21U-jre_x64_windows_hotspot_21.0.13.1_2.zip"
JRE_SHA="e15ee1e54a5a3b38027a09cedfa87bc3348d7dd3ce50866dd3d9671d7fffc6c1"
SYNTHEA_URL="https://github.com/synthetichealth/synthea/releases/download/v4.0.0/synthea-with-dependencies.jar"
if [ ! -d jre ]; then
  curl -sSL -o jre.zip "$JRE_URL"
  echo "$JRE_SHA  jre.zip" | sha256sum -c -
  mkdir -p jre_tmp && unzip -q jre.zip -d jre_tmp && cp -r jre_tmp/*/ jre && rm -rf jre_tmp jre.zip  # cp, not mv: Windows antivirus can lock fresh folders
fi
[ -f synthea-v4.0.0.jar ] || curl -sSL -o synthea-v4.0.0.jar "$SYNTHEA_URL"
sha256sum synthea-v4.0.0.jar
