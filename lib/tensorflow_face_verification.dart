import 'dart:io';
import 'dart:math' as math;
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper_plus/tflite_flutter_helper_plus.dart'
as helper;
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:tflite_flutter_plus/src/bindings/types.dart' as tflp_internal;

class FaceVerification {
  static FaceVerification? _instance;
  static late final Interpreter _interpreter;

  FaceVerification._internal();

  /// Initializes the singleton instance, model => facenet.tfflite
  static Future<void> init({required String modelPath}) async {
    try {
      _interpreter = await Interpreter.fromAsset(modelPath);
      _instance = FaceVerification._internal();
    } catch (e) {
      throw Exception('Failed to load model at "$modelPath": $e');
    }
  }

  /// Check if the service is initialized
  static bool get isInitialized => _instance != null;

  /// Access the instance safely
  static FaceVerification get instance {
    if (!isInitialized) {
      throw Exception(
        'FaceVerificationService not initialized. Call await FaceVerificationService.init() first.',
      );
    }
    return _instance!;
  }

  /// Check is the faces match
  Future<bool> verifySamePerson(
      File? input1,
      File? input2, {
        double threshold = 0.6,
      }) async {
    if(input1 == null){
      throw Exception("File for input 1 not found. Please check the file path and try again.");
    }
    if(input2 == null){
      throw Exception("File for input 2 not found. Please check the file path and try again.");
    }
    final similarity = await getSimilarityScoreFromFile(input1,input2);
    return similarity > threshold;
  }


  /// Compare two face image files and return a similarity score (0 to 1)
  Future<double> getSimilarityScoreFromFile(
      File? input1,
      File? input2,
      ) async {
    if (input1 == null || !input1.existsSync()) {
      throw Exception("File for input 1 not found. Please check the file path and try again.");
    }
    if (input2 == null ||  !input2.existsSync()) {
      throw Exception("File for input 2 not found. Please check the file path and try again.");
    }

    final results = await Future.wait([
      extractFaceRegion(input1),
      extractFaceRegion(input2),
    ]);

    final face1 = results[0];
    final face2 = results[1];

    if (face1 == null || face2 == null) throw Exception("No face detected.");

    return await getSimilarityScoreFromImage(face1, face2);
  }

  /// Compare two faces and return a similarity score (0 to 1)
  Future<double> getSimilarityScoreFromImage(
      image_lib.Image? image1,
      image_lib.Image? image2,
      ) async {
    if (image1 == null) {
      throw Exception("Image 1 not found. Please check the file path and try again.");
    }
    if (image2 == null) {
      throw Exception("Image 2 not found. Please check the file path and try again.");
    }

    final embeddings = await Future.wait([
      extractFaceEmbedding(image1),
      extractFaceEmbedding(image2),
    ]);

    return getSimilarityScore(embeddings[0], embeddings[1]);
  }

  /// Detect and tightly crop the face
  Future<image_lib.Image?> extractFaceRegion(File imageFile) async {
    final inputImage = InputImage.fromFile(imageFile);
    final faceDetector = FaceDetector(
      options: FaceDetectorOptions(performanceMode: FaceDetectorMode.accurate),
    );

    final List<Face> faces = await faceDetector.processImage(inputImage);
    if (faces.isEmpty) return null;

    final face = faces.first;
    final boundingBox = face.boundingBox;

    final bytes = await imageFile.readAsBytes();
    final originalImage = image_lib.decodeImage(bytes);
    if (originalImage == null) return null;

    const double marginRatio = 0.08;
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

  /// Generate embedding vector from cropped image
  Future<List<double>> extractFaceEmbedding(image_lib.Image image) async {
    var inputImage = helper.TensorImage(tflp_internal.TfLiteType.float32);
    inputImage.loadImage(image);

    final processor =
    helper.ImageProcessorBuilder()
        .add(helper.ResizeOp(160, 160, helper.ResizeMethod.bilinear))
        .add(helper.NormalizeOp(127.5, 127.5))
        .build();

    inputImage = processor.process(inputImage);

    var outputBuffer = helper.TensorBufferFloat([1, 128]);
    _interpreter.run(inputImage.buffer, outputBuffer.buffer);

    return outputBuffer.getDoubleList();
  }

  /// Cosine similarity between two embeddings
  double getSimilarityScore(List<double> a, List<double> b) {
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (math.sqrt(normA) * math.sqrt(normB));
  }
}
