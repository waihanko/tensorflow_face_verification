# Flutter Face Verification (Face Embedding & Comparison)
https://github.com/user-attachments/assets/f4248745-b9ac-4990-a3a6-bd8746047b2a

This project provides face verification capabilities using [TensorFlow Lite](https://www.tensorflow.org/lite) and [Google ML Kit Face Detection](https://pub.dev/packages/google_mlkit_face_detection)  
via a lightweight helper class built in Dart.

- 🧠 128D Face Embedding Generation (FaceNet model)
- 🔍 ML Kit Face Detection with auto-cropping
- 📸 Compare faces and compute similarity score

> 🚨 The code assumes the use of a tflite model `facenet.tflite`.

> ⚙️ Works offline — all computation is done on-device.

> ✅ Fully asynchronous and customizable.

---

## 🧪 Features

- Match faces between two images
- Detect and crop the face region from an image
- Measure similarity between faces
- Offline use – no internet required
---

## 🧪 Note
This package is rely on [tflite_flutter](https://pub.dev/packages/tflite_flutter).

🚨 This package may not work in the iOS simulator. It's recommended that you test with a physical device.

🚨 When creating a release archive (IPA), the symbols are stripped by Xcode, so the command `flutter build ipa` may throw a `Failed to lookup symbol ... symbol not found` error. To work around this:

1. In Xcode, go to **Target Runner > Build Settings > Strip Style**
2. Change from **All Symbols** to **Non-Global Symbols**

---


## 🚀 Getting Started

### 1. Add Your Model

Place your model in `assets/models/facenet.tflite`.

```yaml
flutter:
  assets:
    - assets/models/facenet.tflite
```

---

## ⚙️ Usage Example

### Initialize

```dart
await FaceVerification.init(modelPath: 'assets/models/facenet.tflite');
```

### Compare Two Face Images

```dart
bool isSame = await FaceVerification.instance.verifySamePerson(
  File('path/to/image1.jpg'),
  File('path/to/image2.jpg'),
  threshold: 0.6,
);
print("Same person? $isSame");
```

### Get Similarity Score

```dart
double score = await FaceVerification.instance.getFaceSimilarityFromFile(
  File('path/to/image1.jpg'),
  File('path/to/image2.jpg'),
);
print("Similarity: $score");
```

```dart
Image image1 = ....;
Image image2 = ....;

double score = await FaceVerification.instance.getFaceSimilarityFromImage(
  image1,
  image2,
);
print("Similarity: $score");
```

### Additional customizations

```dart
Image image1FaceRegion = await FaceVerification.instance.extractFaceRegion(
  File('path/to/image1.jpg')
);

Image image2FaceRegion = await FaceVerification.instance.extractFaceRegion(
  File('path/to/image1.jpg')
);

List<double> image1EmbeddingVector = await FaceVerification.instance.extractFaceEmbedding(
  image1FaceRegion
);

List<double> image2EmbeddingVector = await FaceVerification.instance.extractFaceEmbedding(
  image2FaceRegion
);

double score = await FaceVerification.instance.getSimilarityScore(
  image1EmbeddingVector, image2EmbeddingVector
);

print("Similarity: $score");
print("Same person ${score > 0.6}");
```

---

## 🤝 Contributing

Feel free to fork, modify, and submit a PR.  
Suggestions, bug reports, or new features are welcome.

---

## 📄 License

MIT License
