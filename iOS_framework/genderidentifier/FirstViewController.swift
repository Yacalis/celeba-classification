//
//  FirstViewController.swift
//  genderidentifier
//
//  Created by Galen Yacalis on 3/8/18.
//  Copyright © 2018 Galen Yacalis. All rights reserved.
//

import UIKit
import AVKit
import Vision
import CoreML

// this first view controller is for predicting classes on images sampled from live video capture
class FirstViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    var label = UILabel()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set up the label that will display prediction results
        label.frame = CGRect(x: 0, y: 575, width: self.view.frame.width, height: 50)
        label.text = "detecting class..."
        label.textAlignment = .center
        label.backgroundColor = UIColor.green
        label.font = UIFont(name: "Futura", size: 20)
        self.view.addSubview(label)
        
        // Note: the below capture session code was based on code from this video:
        // name: CoreML: Real Time Camera Object Detection with Machine Learning - Swift 4
        // url: https://www.youtube.com/watch?v=p6GA8ODlnX0
        
        // start the camera, make sure session can be captured, and start the session
        let captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo
        
        guard let captureDevice = AVCaptureDevice.default(for: .video) else {
            fatalError("could not detect capture device (should be video capture)")
        }
        guard let input = try? AVCaptureDeviceInput(device: captureDevice) else {
            fatalError("could not detect capture device input (should be back camera)")
        }
        captureSession.addInput(input)
        captureSession.startRunning()
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        view.layer.addSublayer(previewLayer)
        previewLayer.frame = view.frame
        
        let dataOutput = AVCaptureVideoDataOutput()
        dataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(dataOutput)
        
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        //image must be converted to cvpixelbuffer to be used as input to model
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            fatalError("could not cast sample buffer as cvpixelbuffer")
        }
        
        // load celeba model specifically as vncoremlmodel, which allows for multilabel classification
        guard let model = try? VNCoreMLModel(for: celeba().model) else{
            fatalError("could not import model as vncoremlmodel")
        }
        
        // create prediction request
        let request = VNCoreMLRequest(model: model) { (finReq, err) in
            
            // need to make sure results can be obtained as vncoremlfeaturevalueobservation;
            // this is specifically because the model is multilabel classification and not
            // a softmax type of classification
            guard let results = finReq.results as? [VNCoreMLFeatureValueObservation] else {
                fatalError("could not get vncoremlrequest finished request as [vnclassificationobservation]")
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
                // If want to see classification probabilities on label, use the below code
                // self?.label.text = "\(multiFeatureArr[0]), \(multiFeatureArr[1]), \(multiFeatureArr[2])"
            }
        }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

