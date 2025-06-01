import 'dart:io';
import 'dart:math' as math;
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:path_provider/path_provider.dart';

class FaceVerification {
  static FaceVerification? _instance;
  static late final Interpreter _interpreter;

  FaceVerification._internal();

  static Future<void> init({required String modelPath}) async {
    try {
      _interpreter = await Interpreter.fromAsset(modelPath);
      _instance = FaceVerification._internal();
    } catch (e) {
      throw Exception('Failed to load model at "$modelPath": $e');
    }
  }

  static bool get isInitialized => _instance != null;

  static FaceVerification get instance {
    if (!isInitialized) {
      throw Exception(
        'FaceVerificationService not initialized. Call await FaceVerificationService.init() first.',
      );
    }
    return _instance!;
  }

  Future<bool> verifySamePerson(
      File? input1,
      File? input2, {
        double threshold = 0.6,
      }) async {
    if (input1 == null) {
      throw Exception("File for input 1 not found.");
    }
    if (input2 == null) {
      throw Exception("File for input 2 not found.");
    }

    final similarity = await getSimilarityScoreFromFile(input1, input2);
    return similarity > threshold;
  }

  Future<double> getSimilarityScoreFromFile(
      File input1,
      File input2,
      ) async {
    final results = await Future.wait([
      extractFaceRegion(input1),
      extractFaceRegion(input2),
    ]);

    final face1 = results[0];
    final face2 = results[1];

    if (face1 == null || face2 == null) {
      throw Exception("No face detected.");
    }

    return await getSimilarityScoreFromImage(face1, face2);
  }

  Future<double> getSimilarityScoreFromImage(
      image_lib.Image image1,
      image_lib.Image image2,
      ) async {
    final embeddings = await Future.wait([
      extractFaceEmbedding(image1),
      extractFaceEmbedding(image2),
    ]);

    return getSimilarityScore(embeddings[0], embeddings[1]);
  }

  Future<image_lib.Image?> extractFaceRegion(File imageFile) async {
    InputImage inputImage;
    final faceDetector = FaceDetector(
      options: FaceDetectorOptions(performanceMode: FaceDetectorMode.accurate),
    );

    if (Platform.isIOS) {
      final File iosImageProcessed =
      await bakeImageOrientation(imageFile);
      inputImage = InputImage.fromFilePath(iosImageProcessed.path);
    } else {
      inputImage = InputImage.fromFile(imageFile);
    }

    final faces = await faceDetector.processImage(inputImage);
    if (faces.isEmpty) return null;

    final face = faces.first;
    final boundingBox = face.boundingBox;

    final bytes = await imageFile.readAsBytes();
    final originalImage = image_lib.decodeImage(bytes);
    if (originalImage == null) return null;

    const marginRatio = 0.08;
    final int x = (boundingBox.left - boundingBox.width * marginRatio)
        .toInt()
        .clamp(0, originalImage.width - 1);
    final int y = (boundingBox.top - boundingBox.height * marginRatio)
        .toInt()
        .clamp(0, originalImage.height - 1);
    final int w = (boundingBox.width * (1 + 2 * marginRatio)).toInt();
    final int h = (boundingBox.height * (1 + 2 * marginRatio)).toInt();

    int size = math.max(w, h);
    size = math.min(
      size,
      math.min(originalImage.width - x, originalImage.height - y),
    );

    final cropped = image_lib.copyCrop(originalImage, x, y, size, size);
    final resized = image_lib.copyResize(cropped, width: 160, height: 160);

    return resized;
  }

  Future<List<double>> extractFaceEmbedding(image_lib.Image image) async {
    // Resize and normalize the image
    final input = List.generate(1, (_) => List.generate(160, (y) {
      return List.generate(160, (x) {
        final pixel = image.getPixel(x, y);
        final r = ((image_lib.getRed(pixel)) - 127.5) / 127.5;
        final g = ((image_lib.getGreen(pixel)) - 127.5) / 127.5;
        final b = ((image_lib.getBlue(pixel)) - 127.5) / 127.5;
        return [r, g, b];
      });
    }));

    final output = List.filled(128, 0.0).reshape([1, 128]);
    _interpreter.run(input, output);

    return List<double>.from(output[0]);
  }

  double getSimilarityScore(List<double> a, List<double> b) {
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dot / (math.sqrt(normA) * math.sqrt(normB));
  }

  bakeImageOrientation(File pickedFile) async {
    if (Platform.isIOS) {
      final directory = await getApplicationDocumentsDirectory();
      final path = directory.path;
      final filename = DateTime.now().millisecondsSinceEpoch.toString();

      final image_lib.Image? capturedImage =
      image_lib.decodeImage(await pickedFile.readAsBytes());

      final image_lib.Image orientedImage = image_lib.bakeOrientation(capturedImage!);

      final imageToBeProcessed = await File('$path/$filename')
          .writeAsBytes(image_lib.encodeJpg(orientedImage));

      return imageToBeProcessed;
    }
    return null;
  }

}
