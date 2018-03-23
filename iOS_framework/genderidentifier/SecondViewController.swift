//
//  SecondViewController.swift
//  genderidentifier
//
//  Created by Galen Yacalis on 3/8/18.
//  Copyright © 2018 Galen Yacalis. All rights reserved.
//

/**
 * Copyright (c) 2017 Razeware LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * Notwithstanding the foregoing, you may not use, copy, modify, merge, publish,
 * distribute, sublicense, create a derivative work, and/or sell copies of the
 * Software in any work that is designed, intended, or marketed for pedagogical or
 * instructional purposes related to programming, coding, application development,
 * or information technology.  Permission for such use, copying, modification,
 * merger, publication, distribution, sublicensing, creation of derivative works,
 * or sale is expressly withheld.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

import UIKit
import CoreML
import Vision

// this second view controller is for predicting classes on images selected from the photo library
// most of image selection code is written by Razeware LLC
// label properties, button properties, image properties, and model code is written by Galen Yacalis
class SecondViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    // MARK: - Properties
    var label = UILabel()
    var scene = UIImageView()
    var button = UIButton()
    
    // MARK: - View Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // set label properties
        label.frame = CGRect(x: 0, y: 575, width: self.view.frame.width, height: 50)
        label.text = "detecting class..."
        label.textAlignment = .center
        label.backgroundColor = UIColor.green
        label.font = UIFont(name: "Futura", size: 20)
        self.view.addSubview(label)
        
        //set button properties
        button.frame = CGRect(x: 0, y: 525, width: self.view.frame.width, height: 50)
        button.setTitle("Select new image", for: .normal)
        button.backgroundColor = UIColor.red
        button.tintColor = UIColor.black
        button.addTarget(self, action: #selector(self.pickImage(_:)), for: .touchUpInside)
        self.view.addSubview(button)
        
        // set up default image
        guard let image = UIImage(named: "train_night") else {
            fatalError("no starting image")
        }
        scene.image = image
        let maxHeight = self.view.frame.height - 175
        let maxWidth = self.view.frame.width
        let sceneWidth = image.size.width < maxWidth ? image.size.width : maxWidth
        let sceneHeight = image.size.height < maxHeight ? image.size.height : maxHeight
        scene.frame = CGRect(x: 0, y: 20, width: sceneWidth, height: sceneHeight)
        self.view.addSubview(scene)
        
        // detect the scene of the default image
        guard let ciImage = CIImage(image: image) else {
            fatalError("couldn't convert UIImage to CIImage")
        }
        detectScene(image: ciImage)

    }
    
    // MARK - Functions
    func detectScene(image: CIImage) {
        
        label.text = "detecting scene..."
        
        // Load the ML model through its generated class
        guard let model = try? VNCoreMLModel(for: celeba().model) else {
            fatalError("can't load model")
        }
        
        // Create a Vision request with completion handler
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            
            // need to make sure results can be obtained as vncoremlfeaturevalueobservation;
            // this is specifically because the model is multilabel classification and not
            // a softmax type of classification
            guard let results = request.results as? [VNCoreMLFeatureValueObservation] else {
                    fatalError("unexpected result type from VNCoreMLRequest")
            }
            guard let multiFeatureArr = results.first?.featureValue.multiArrayValue else {
                fatalError("no multiFeatureArr")
            }
            
            // determine if probabilities returned by model are closer to 1 or 0 for each class
            let glasses = Double(multiFeatureArr[0]) > 0.5 ? "glasses" : "no glasses"
            let male = Double(multiFeatureArr[1]) > 0.5 ? "male" : "female"
            let smiling = Double(multiFeatureArr[2]) > 0.5 ? "smiling" : "not smiling"
            // update UI on main queue
            DispatchQueue.main.async { [weak self] in
                self?.label.text = "\(glasses), \(male), \(smiling)"
            }
        }
        
        // need a handler
        let handler = VNImageRequestHandler(ciImage: image)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request])
            } catch {
                print(error)
            }
        }
    }
    
    // more UI button properties
    @objc func pickImage(_ sender: UIButton) {
        let pickerController = UIImagePickerController()
        pickerController.delegate = self
        pickerController.sourceType = .savedPhotosAlbum
        self.present(pickerController, animated: true, completion: nil)
    }
    
    // MARK: - UIImagePickerControllerDelegate
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        
        // replace scene's image with new image that was picked
        guard let image = info[UIImagePickerControllerOriginalImage] as? UIImage else {
            fatalError("couldn't load image from Photos")
        }
        scene.image = image
        
        // detect the scene of the new image
        guard let ciImage = CIImage(image: image) else {
            fatalError("couldn't convert UIImage to CIImage")
        }
        detectScene(image: ciImage)
        dismiss(animated: true)
        
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
}
