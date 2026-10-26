// Apple Vision and Core Image helpers for the OCR bake-off. Everything here is
// part of macOS; nothing is downloaded.
//
//   vision_tool ocr     <image> <out.json>          L4:  full-page document recognition
//   vision_tool ocr-raw <image> <out.json>          L4b: same, language correction OFF
//   vision_tool blocks  <image> <out.json>          L2b: text-block boxes only
//   vision_tool clean   <in.png> <out.png>          L5c: deskew + grayscale + sharpen
//   vision_tool degrade <in.png> <out.jpg> <seed>   Tier A simulated scan
//
// Boxes are written normalised to [0,1] with a TOP-LEFT origin, as
// [x0, y0, x1, y1], so Python never has to know Vision's lower-left convention.
// Nothing is printed except errors and a one-line count.

import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

let languages = [Locale.Language(identifier: "de-DE"), Locale.Language(identifier: "en-US"), Locale.Language(identifier: "tr-TR")]

func topLeftBox(_ rect: NormalizedRect) -> [Double] {
    let r = rect.cgRect
    return [Double(r.minX), Double(1 - r.maxY), Double(r.maxX), Double(1 - r.minY)]
}

// Language correction rewrites unlikely strings, which is exactly what an IBAN or an
// invoice number looks like to a language model. L4 keeps it on (Apple's default
// behaviour), L4b turns it off, and the pair measures what it costs in numbers.
func documentRequest(correction: Bool = true) -> RecognizeDocumentsRequest {
    var request = RecognizeDocumentsRequest()
    request.textRecognitionOptions.recognitionLanguages = languages
    request.textRecognitionOptions.automaticallyDetectLanguage = false
    request.textRecognitionOptions.useLanguageCorrection = correction
    request.barcodeDetectionOptions.enabled = false
    return request
}

func writeJSON(_ object: Any, to path: String) throws {
    let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: path), options: .atomic)
}

func runOCR(_ input: String, _ output: String, correction: Bool) async throws {
    let started = Date()
    let observations = try await documentRequest(correction: correction).perform(on: URL(fileURLWithPath: input))
    var transcripts: [String] = []
    var words: [[String: Any]] = []
    var lines: [[String: Any]] = []
    for observation in observations {
        let text = observation.document.text
        transcripts.append(text.transcript)
        for line in text.lines {
            lines.append(["text": line.transcript, "box": topLeftBox(line.boundingBox), "confidence": Double(line.confidence)])
        }
        for word in text.words ?? [] {
            words.append(["text": word.transcript, "box": topLeftBox(word.boundingBox)])
        }
    }
    try writeJSON([
        "transcript": transcripts.joined(separator: "\n\n"),
        "lines": lines,
        "words": words,
        "seconds": Date().timeIntervalSince(started),
        "language_correction": correction,
    ], to: output)
    print("ocr ok lines=\(lines.count) words=\(words.count) correction=\(correction)")
}

func runBlocks(_ input: String, _ output: String) async throws {
    let started = Date()
    let observations = try await documentRequest().perform(on: URL(fileURLWithPath: input))
    var blocks: [[String: Any]] = []
    for observation in observations {
        let doc = observation.document
        if let title = doc.title {
            blocks.append(["kind": "title", "box": topLeftBox(title.boundingRegion.boundingBox)])
        }
        for paragraph in doc.paragraphs {
            blocks.append(["kind": "paragraph", "box": topLeftBox(paragraph.boundingRegion.boundingBox)])
        }
        for table in doc.tables {
            blocks.append(["kind": "table", "box": topLeftBox(table.boundingRegion.boundingBox)])
        }
        for list in doc.lists {
            blocks.append(["kind": "list", "box": topLeftBox(list.boundingRegion.boundingBox)])
        }
    }
    try writeJSON(["blocks": blocks, "seconds": Date().timeIntervalSince(started)], to: output)
    print("blocks ok count=\(blocks.count)")
}

// L5c: what OCRmyPDF would do to a scan before Tesseract sees it. Tesseract is fed a
// raw render today, which is the unfair version of the "Tesseract cannot extract"
// claim. Deskew uses the angle Vision reports for the text lines it found, so this
// needs no download either; it is an approximation of Leptonica's deskew, not a copy.
func pageSkewRadians(_ input: String) async -> Double {
    guard let source = CIImage(contentsOf: URL(fileURLWithPath: input)) else { return 0 }
    let width = source.extent.width, height = source.extent.height
    guard let observations = try? await documentRequest().perform(on: URL(fileURLWithPath: input)) else { return 0 }
    var angles: [Double] = []
    for observation in observations {
        for line in observation.document.text.lines {
            // Normalised points distort angles, so measure in pixels.
            let dx = Double(line.topRight.x - line.topLeft.x) * Double(width)
            let dy = Double(line.topRight.y - line.topLeft.y) * Double(height)
            if abs(dx) > Double(width) * 0.1 { angles.append(atan2(dy, dx)) }
        }
    }
    guard angles.count >= 3 else { return 0 }
    let sorted = angles.sorted()
    let median = sorted[sorted.count / 2]
    // Only correct a plausible scanner skew; anything larger is layout, not skew.
    return abs(median) < 5.0 * .pi / 180 ? median : 0
}

func runClean(_ input: String, _ output: String) async throws {
    guard let source = CIImage(contentsOf: URL(fileURLWithPath: input)) else {
        throw NSError(domain: "clean", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot read input image"])
    }
    let skew = await pageSkewRadians(input)
    let extent = source.extent
    var image = source
    if skew != 0 {
        let center = CGPoint(x: extent.midX, y: extent.midY)
        let rotation = CGAffineTransform(translationX: center.x, y: center.y)
            .rotated(by: CGFloat(-skew))
            .translatedBy(x: -center.x, y: -center.y)
        let white = CIImage(color: .white).cropped(to: extent)
        image = image.transformed(by: rotation).composited(over: white).cropped(to: extent)
    }
    image = image.applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0.0, kCIInputContrastKey: 1.15])
    image = image.applyingFilter("CIUnsharpMask", parameters: [kCIInputRadiusKey: 1.2, kCIInputIntensityKey: 0.6])
    let context = CIContext()
    let colorSpace = CGColorSpace(name: CGColorSpace.genericGrayGamma2_2)!
    try context.writePNGRepresentation(of: image, to: URL(fileURLWithPath: output), format: .L8, colorSpace: colorSpace)
    print("clean ok skew_deg=\(String(format: "%.2f", skew * 180 / .pi))")
}

// Deterministic per seed, so a degraded variant can be rebuilt byte-for-byte.
struct SeededGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed &* 6364136223846793005 &+ 1442695040888963407 }
    mutating func next() -> Double {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Double(state >> 11) / Double(1 << 53)
    }
}

func runDegrade(_ input: String, _ output: String, _ seed: UInt64) throws {
    guard let source = CIImage(contentsOf: URL(fileURLWithPath: input)) else {
        throw NSError(domain: "degrade", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot read input image"])
    }
    var rng = SeededGenerator(seed: seed)
    let extent = source.extent

    // 300 DPI render down to 200 DPI.
    var image = source.applyingFilter("CILanczosScaleTransform", parameters: [kCIInputScaleKey: 200.0 / 300.0, kCIInputAspectRatioKey: 1.0])

    // 0.5-1.5 degrees, either direction, on a white page.
    let degrees = (0.5 + rng.next()) * (rng.next() < 0.5 ? -1 : 1)
    let scaled = image.extent
    let center = CGPoint(x: scaled.midX, y: scaled.midY)
    let rotation = CGAffineTransform(translationX: center.x, y: center.y)
        .rotated(by: CGFloat(degrees * .pi / 180))
        .translatedBy(x: -center.x, y: -center.y)
    let white = CIImage(color: .white).cropped(to: scaled)
    image = image.transformed(by: rotation).composited(over: white).cropped(to: scaled)

    // Grayscale, slightly lifted blacks and reduced contrast, like a cheap scanner.
    image = image.applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0.0, kCIInputContrastKey: 0.85, kCIInputBrightnessKey: 0.03])

    // Mild noise from a seed-shifted window of Core Image's fixed random field.
    let offset = CGAffineTransform(translationX: CGFloat(rng.next() * 4096), y: CGFloat(rng.next() * 4096))
    let noise = CIFilter(name: "CIRandomGenerator")!.outputImage!
        .transformed(by: offset)
        .cropped(to: scaled)
        .applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0.08, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0.08, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0.08, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBiasVector": CIVector(x: -0.04, y: -0.04, z: -0.04, w: 1),
        ])
    image = noise.applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: image]).cropped(to: scaled)

    let context = CIContext()
    let colorSpace = CGColorSpace(name: CGColorSpace.genericGrayGamma2_2)!
    try context.writeJPEGRepresentation(
        of: image, to: URL(fileURLWithPath: output), colorSpace: colorSpace,
        options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.6])
    print("degrade ok from=\(Int(extent.width))x\(Int(extent.height)) to=\(Int(scaled.width))x\(Int(scaled.height))")
}

let args = CommandLine.arguments
do {
    switch (args.count > 1 ? args[1] : "", args.count) {
    case ("ocr", 4): try await runOCR(args[2], args[3], correction: true)
    case ("ocr-raw", 4): try await runOCR(args[2], args[3], correction: false)
    case ("blocks", 4): try await runBlocks(args[2], args[3])
    case ("clean", 4): try await runClean(args[2], args[3])
    case ("degrade", 5): try runDegrade(args[2], args[3], UInt64(args[4]) ?? 0)
    default:
        FileHandle.standardError.write("usage: vision_tool ocr|ocr-raw|blocks <image> <out.json> | clean <in> <out.png> | degrade <in> <out.jpg> <seed>\n".data(using: .utf8)!)
        exit(2)
    }
} catch {
    FileHandle.standardError.write("vision_tool error: \(error.localizedDescription)\n".data(using: .utf8)!)
    exit(1)
}
