import 'package:flutter/material.dart';
import 'package:tensorflow_face_verification/tensorflow_face_verification.dart';

import 'face_compare_screen.dart';

void main() async{
  WidgetsFlutterBinding.ensureInitialized();
  await FaceVerification.init(modelPath: "assets/models/facenet.tflite");
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const FaceCompareScreen(),
    );
  }
}


