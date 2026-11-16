"""Draw Transcriber's app icon: a white waveform on a deep blue rounded square.

Usage: uv run make_icon.py OUT.icns
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import AppKit


def render(size):
    rep = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, AppKit.NSDeviceRGBColorSpace, 0, 0
    )
    AppKit.NSGraphicsContext.saveGraphicsState()
    AppKit.NSGraphicsContext.setCurrentContext_(
        AppKit.NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    )

    # macOS icon grid: the shape fills ~80% of the canvas.
    inset = size * 0.1
    body = ((inset, inset), (size - 2 * inset, size - 2 * inset))
    shape = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(body, size * 0.18, size * 0.18)
    top = AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.20, 0.42, 0.98, 1)
    bottom = AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.10, 0.20, 0.62, 1)
    AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(top, bottom).drawInBezierPath_angle_(shape, -90)

    config = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
        size * 0.36, AppKit.NSFontWeightSemibold
    ).configurationByApplyingConfiguration_(
        AppKit.NSImageSymbolConfiguration.configurationWithPaletteColors_([AppKit.NSColor.whiteColor()])
    )
    symbol = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
        "waveform", None
    ).imageWithSymbolConfiguration_(config)
    w, h = symbol.size()
    symbol.drawInRect_((((size - w) / 2, (size - h) / 2), (w, h)))

    AppKit.NSGraphicsContext.restoreGraphicsState()
    return rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {})


def main(out):
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for points in (16, 32, 128, 256, 512):
            render(points).writeToFile_atomically_(str(iconset / f"icon_{points}x{points}.png"), True)
            render(points * 2).writeToFile_atomically_(str(iconset / f"icon_{points}x{points}@2x.png"), True)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", out], check=True)


if __name__ == "__main__":
    main(sys.argv[1])
