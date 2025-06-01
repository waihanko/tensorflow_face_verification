import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tensorflow_face_verification/tensorflow_face_verification.dart';

class FaceCompareScreen extends StatefulWidget {
  const FaceCompareScreen({super.key});

  @override
  _FaceCompareScreenState createState() => _FaceCompareScreenState();
}

class _FaceCompareScreenState extends State<FaceCompareScreen> {
  late Interpreter interpreter;
  File? image1;
  File? image2;
  bool isProcessImage1 = false;
  bool isProcessImage2 = false;
  final faceService = FaceVerification.instance;
  bool isLoading = false;
  @override
  void initState() {
    super.initState();
  }

  void _pickImage(int imageNumber) async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) {
      setState(() {
        isProcessImage1 = false;
        isProcessImage2 = false;
      });
      return;
    }

    setState(() {
      if (imageNumber == 1) {
        setState(() {
          image1 = File(picked.path);
          isProcessImage1 = false;
        });
      } else {
        setState(() {
          image2 = File(picked.path);
          isProcessImage2 = false;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Face Validation and Verification')),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if(isLoading) CircularProgressIndicator(),
          SizedBox(height: 48,),
          Row(
            mainAxisSize: MainAxisSize.max,
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _buildImagePicker(
                  onTap: () {
                    setState(() {
                      image1 = null;
                      isProcessImage1 = true;
                    });
                    _pickImage(1);
                  },
                  image: image1,
                  icon: Icons.add_a_photo,
                  isProcessImage: isProcessImage1),
              _buildImagePicker(
                onTap: () {
                  setState(() {
                    image2 = null;
                    isProcessImage2 = true;
                  });
                  _pickImage(2);
                },
                image: image2,
                icon: Icons.add_a_photo,
                isProcessImage: isProcessImage2,
              ),
            ],
          ),
          const SizedBox(
            height: 20,
          ),
          VerifyButtonWidget(
            buttonText: "Verify Same Person",
            onPress: () async {
              setState(() {
                isLoading = true;
              });
              bool isTheSamePerson =
                  await faceService.verifySamePerson(image1, image2, threshold: 0.6);
              setState(() {
                isLoading = false;
              });

              showDialog(
                context: context,
                builder: (_) => AlertDialog(
                  title: const Text('Validation Result'),
                  content: Text(
                    'Is the same person => $isTheSamePerson',
                    style: const TextStyle(fontSize: 18),
                  ),
                ),
              );
            },
          ),
          const SizedBox(
            height: 20,
          ),
          VerifyButtonWidget(
            buttonText: "Get Face Similarity",
            onPress: () async {
              setState(() {
                isLoading = true;
              });
              double similarityPoint =
              await faceService.getSimilarityScoreFromFile(image1!, image2!);
              setState(() {
                isLoading = false;
              });
              showDialog(
                context: context,
                builder: (_) => AlertDialog(
                  title: const Text('Validation Result'),
                  content: Text(
                    'Similarity Point => $similarityPoint',
                    style: const TextStyle(fontSize: 18),
                  ),
                ),
              );
            },
          ),
          const SizedBox(
            height: 12,
          ),
          ElevatedButton(
            onPressed: () {
              setState(() {
                image1 = null;
                image2 = null;
              });
            },
            child: Text("Clear"),
          ),
        ],
      ),
    );
  }

  Widget _buildImagePicker({
    required Function onTap,
    required File? image,
    required IconData icon,
    bool isProcessImage = false,
  }) {
    return GestureDetector(
      onTap: () => onTap.call(),
      child: Container(
        width: 150,
        height: 150,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey),
          borderRadius: BorderRadius.circular(10),
        ),
        child: isProcessImage
            ? const SizedBox(
                width: 50,
                height: 50,
                child: Center(child: CircularProgressIndicator()))
            : image == null
                ? Icon(icon, size: 50, color: Colors.grey)
                : ClipRRect(
                    borderRadius: BorderRadius.circular(10),
                    child: Image.file(
                      image,
                      fit: BoxFit.cover,
                    ),
                  ),
      ),
    );
  }
}

class VerifyButtonWidget extends StatelessWidget {
  final Function onPress;
  final String buttonText;

  const VerifyButtonWidget({
    required this.onPress,
    required this.buttonText,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: ()=> onPress(),
      style: ElevatedButton.styleFrom(
        padding: EdgeInsets.symmetric(horizontal: 24, vertical: 14),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
        elevation: 4,
      ),
      child: Text(
        buttonText,
        style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
      ),
    );
  }
}
