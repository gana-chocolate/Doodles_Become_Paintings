//
//  mainViewController.swift
//  Capstone
//
//  Created by Kim Chan Il on 30/04/2019.
//  Copyright © 2019 Kim Chan Il. All rights reserved.
//

import UIKit
import SnapKit
import Canvas_
import Alamofire
import AlamofireImage
import Foundation
import Photos




class mainViewController: UIViewController, CanvasEvents, UIPickerViewDelegate, UIPickerViewDataSource {
    // 생성할 컴포넌트의 개수 정의
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    // 컴포넌트가 가질 목록의 길이
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return 2
    }
    
    @IBOutlet weak var imageView: UIImageView!
    // 컴포넌트의 목록 각 행에 출력될 내용
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        

        if row == 0 {
            imageID = "99999"
            return "car"
        }
        else {
            imageID = "11111"

            return "bicycle"
        }
    }
    
    // 컴포넌트의 행을 선택했을 때 실행할 액션
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        
    }
    
    @IBOutlet weak var mainTitle: UITextView!
    @IBOutlet weak var modelPicker: UIPickerView!
    @IBOutlet weak var button: UIButton!
    @IBOutlet weak var sketchView: UIView!
    
    
    
    
    
    
    var imageID = ""
    
    @objc
    func submit(){
    
    
        
    }
    
    @objc
    func download(){
        sketchView.isHidden = true
        undoBtn.isHidden = true
        redoBtn.isHidden = true
        
        imageView.isHidden = false
        saveBtn.isHidden = false
        backBtn.isHidden = false
        sketchView.setNeedsDisplay()
        sketchView.layoutIfNeeded()
        
        
        let url = "http://220.71.91.185:8888/mainapp/Upload/" /* your API url */
        
        let headers: HTTPHeaders = [
            "Cookie": "",
            "Content-type": "multipart/form-data"
           
        ]
        
        Alamofire.upload(multipartFormData: { (multipartFormData) in
      
            
            multipartFormData.append((self.canvas.export().pngData()?.base64EncodedData(options: .lineLength64Characters))!, withName: "imgBase64" as String)
            
            
            
            multipartFormData.append(self.canvas.export().jpegData(compressionQuality: 1)!, withName: "uploadfile", fileName: self.imageID+".jpg", mimeType: "uploadfile/jpg")
            
            
        }, usingThreshold: UInt64.init(), to: url, method: .post, headers: headers) { (result) in
            switch result{
            case .success(let upload, _, _):
                upload.responseJSON { response in
                    print("Succesfully uploaded")
                    if let err = response.error{
                        //onError?(err)
                        return
                    }
                    //onCompletion?(nil)
                }
            case .failure(let error):
                print("Error in upload: \(error.localizedDescription)")
                //onError?(error)
            }
        }
            
        sleep(20)
        
        Alamofire.request("http://220.71.91.185:8888/mainapp/media/"+imageID+".jpg").responseImage { response in
            debugPrint(response)
            
            print(response.request)
            print(response.response)
            debugPrint(response.result)
            
            if let image = response.result.value {
                self.imageView.image = image
                print("image downloaded: \(image)")
            }
        }
 

    }
    
    lazy var canvas: Canvas = {
        
        let a = Canvas(createDefaultLayer: true)
        a.translatesAutoresizingMaskIntoConstraints = false
        a.backgroundColor = .white
        a.layer.masksToBounds = true
        a.layer.cornerRadius = 20
        a.delegate = self
        
        return a
    }()
    lazy var undoBtn: UIButton = {
        let a = UIButton()
        a.translatesAutoresizingMaskIntoConstraints = false
        a.setTitle("Undo", for: .normal)
        a.backgroundColor = UIColor.lightGray
        
        a.addTarget(self, action: #selector(undo), for: .touchUpInside)
        
        return a
    }()
    
    lazy var redoBtn: UIButton = {
        let a = UIButton()
        a.translatesAutoresizingMaskIntoConstraints = false
        a.setTitle("Redo", for: .normal)
        a.backgroundColor = UIColor.gray
        a.addTarget(self, action: #selector(redo), for: .touchUpInside)
        
        return a
    }()
    lazy var plusBtn: UIButton = {
        let a = UIButton()
        a.translatesAutoresizingMaskIntoConstraints = false
        a.setTitle("+", for: .normal)
        a.backgroundColor = UIColor.gray
        a.addTarget(self, action: #selector(addPoint), for: .touchUpInside)
        
        return a
    }()
    lazy var minusBtn: UIButton = {
        let a = UIButton()
        a.translatesAutoresizingMaskIntoConstraints = false
        a.setTitle("-", for: .normal)
        a.backgroundColor = UIColor.gray
        a.addTarget(self, action: #selector(minusPoint), for: .touchUpInside)
        
        return a
    }()
    @objc
    func addPoint() {
        self.canvas.currentBrush.thickness = self.canvas.currentBrush.thickness + 3
    }
    @objc
    func minusPoint() {
        self.canvas.currentBrush.thickness = self.canvas.currentBrush.thickness - 3
    }
    
    lazy var saveBtn: UIButton = {
        let a = UIButton()
        a.translatesAutoresizingMaskIntoConstraints = false
        a.setTitle("SAVE", for: .normal)
        a.backgroundColor = UIColor.gray
        a.addTarget(self, action: #selector(redo), for: .touchUpInside)
        
        return a
    }()
    
    
    func saveImageToCameraRoll(image: UIImage) {
        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAsset(from: image)
        }, completionHandler: { success, error in
            if success {
                print("save success")
            }
            else if let error = error {
                print("save error \(error)")
            }
            else {
                print("save ok")
            }
        })
    }
    
    
    
    
    
    @objc
    func save(){
        saveImageToCameraRoll(image: self.imageView.image!)
        let alertController = UIAlertController(title: "사진 저장", message: "생성된 사진이 저장되었습니다.", preferredStyle: .alert)
        
        let okAction = UIAlertAction(title: "확인", style: UIAlertAction.Style.default)
        alertController.addAction(okAction)
        present(alertController, animated: true, completion: nil)
    }
    lazy var backBtn:UIButton = {
        let a = UIButton()
        a.translatesAutoresizingMaskIntoConstraints = false
        a.setTitle("BACK", for: .normal)
        a.backgroundColor = UIColor.gray
        a.addTarget(self, action: #selector(redo), for: .touchUpInside)
        
        return a
    }()
    
    @objc
    func back(){
        sketchView.isHidden = false
        undoBtn.isHidden = false
        redoBtn.isHidden = false
        canvas.clear()
        imageView.isHidden = true
        saveBtn.isHidden = true
        backBtn.isHidden = true
        sketchView.setNeedsDisplay()
        sketchView.layoutIfNeeded()
    }
    
    
    override func viewDidLoad() {
        super.viewDidLoad()

        sketchView.translatesAutoresizingMaskIntoConstraints = false
        sketchView.backgroundColor = .brown
        modelPicker.delegate = self

        
        setViewDetail()
        layoutbutton()

    }
    func setViewDetail(){

        sketchView.addSubview(canvas)
        view.addSubview(undoBtn)
        view.addSubview(redoBtn)
        view.addSubview(imageView)
        view.addSubview(sketchView)
        view.addSubview(saveBtn)
        view.addSubview(backBtn)
        view.addSubview(plusBtn)
        view.addSubview(minusBtn)
        saveBtn.isHidden = true
        backBtn.isHidden = true
        imageView.isHidden = true
        redoBtn.isHidden = false
        undoBtn.isHidden = false

        sketchView.layer.masksToBounds = false
        sketchView.layer.cornerRadius = 20
        sketchView.layer.borderWidth = 2
        sketchView.layer.borderColor = UIColor.cyan.withAlphaComponent(120/255).cgColor
        sketchView.layer.shadowColor = UIColor.black.cgColor
        sketchView.layer.shadowRadius = 2
        sketchView.layer.shadowOpacity = 0.5
        sketchView.layer.shadowOffset = CGSize(width: 0, height: 0)
        
        
        button.setTitle("SUBMIT", for: .normal)
        button.tintColor = UIColor.white

        button.backgroundColor = UIColor.cyan.withAlphaComponent(120/255)
        button.layer.cornerRadius = 5
        button.layer.borderWidth = 2
        button.layer.borderColor = UIColor.cyan.withAlphaComponent(120/255).cgColor
        button.layer.shadowColor = UIColor.black.cgColor
        button.layer.shadowRadius = 2
        button.layer.shadowOpacity = 0.5
        button.layer.shadowOffset = CGSize(width: 0, height: 0)
        
        mainTitle.text = "Doodles Become Paintings"
        mainTitle.isUserInteractionEnabled = false
        mainTitle.font = UIFont.systemFont(ofSize: 60, weight: .heavy)
        mainTitle.textColor = UIColor(red: 250/255, green: 128/255, blue: 114/255, alpha: 1)
        
        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(zoom(sender:)))
        view.addGestureRecognizer(pinch)
        canvas.currentBrush.thickness = 35
        button.addTarget(self, action: #selector(download), for: .touchUpInside)
        backBtn.addTarget(self, action: #selector(back), for: .touchUpInside)
        saveBtn.addTarget(self, action: #selector(save), for: .touchUpInside)
    }

    func layoutbutton(){
        canvas.centerXAnchor.constraint(equalTo: sketchView.centerXAnchor).isActive = true
        canvas.topAnchor.constraint(equalTo: sketchView.topAnchor).isActive = true
        canvas.widthAnchor.constraint(equalTo: sketchView.widthAnchor).isActive = true
        canvas.heightAnchor.constraint(equalTo: sketchView.heightAnchor).isActive = true
        

        undoBtn.topAnchor.constraint(equalTo: sketchView.bottomAnchor).isActive = true
        undoBtn.leadingAnchor.constraint(equalTo: sketchView.leadingAnchor).isActive = true
        undoBtn.widthAnchor.constraint(equalTo: sketchView.widthAnchor, multiplier: 0.5).isActive = true
        undoBtn.heightAnchor.constraint(equalToConstant: 40).isActive = true
        
        redoBtn.topAnchor.constraint(equalTo: sketchView.bottomAnchor).isActive = true
        redoBtn.leadingAnchor.constraint(equalTo: undoBtn.trailingAnchor).isActive = true
        redoBtn.widthAnchor.constraint(equalTo: sketchView.widthAnchor, multiplier: 0.5).isActive = true
        redoBtn.heightAnchor.constraint(equalToConstant: 40).isActive = true
        

        saveBtn.topAnchor.constraint(equalTo: sketchView.bottomAnchor).isActive = true
        saveBtn.leadingAnchor.constraint(equalTo: sketchView.leadingAnchor).isActive = true
        saveBtn.widthAnchor.constraint(equalTo: sketchView.widthAnchor, multiplier: 0.5).isActive = true
        saveBtn.heightAnchor.constraint(equalToConstant: 40).isActive = true
        
        backBtn.topAnchor.constraint(equalTo: sketchView.bottomAnchor).isActive = true
        backBtn.leadingAnchor.constraint(equalTo: undoBtn.trailingAnchor).isActive = true
        backBtn.widthAnchor.constraint(equalTo: sketchView.widthAnchor, multiplier: 0.5).isActive = true
        backBtn.heightAnchor.constraint(equalToConstant: 40).isActive = true
        
        plusBtn.topAnchor.constraint(equalTo: mainTitle.bottomAnchor).isActive = true
        plusBtn.leadingAnchor.constraint(equalTo: sketchView.leadingAnchor).isActive = true
        plusBtn.widthAnchor.constraint(equalTo: sketchView.widthAnchor, multiplier: 0.5).isActive = true
        plusBtn.heightAnchor.constraint(equalToConstant: 40).isActive = true
        
        minusBtn.topAnchor.constraint(equalTo: mainTitle.bottomAnchor).isActive = true
        minusBtn.leadingAnchor.constraint(equalTo: undoBtn.trailingAnchor).isActive = true
        minusBtn.widthAnchor.constraint(equalTo: sketchView.widthAnchor, multiplier: 0.5).isActive = true
        minusBtn.heightAnchor.constraint(equalToConstant: 40).isActive = true
        
    }
    
    ////////////////
    ////////////////
    ////Delegates////
    ////////////////
    ////////////////
    
    func willBeginDrawing(on canvas: Canvas) {
        
    }
    
    func isDrawing(on canvas: Canvas) {
        
    }
    
    func didFinishDrawing(on canvas: Canvas) {
        
    }
    
    func didSampleColor(on canvas: Canvas, sampledColor color: UIColor) {
        
    }
    
    func didPaintNodes(on canvas: Canvas, nodes: [Node], strokeColor: UIColor, fillColor: UIColor?) {
        
    }
    
    func didUndo(on canvas: Canvas) {
        
    }
    
    func didRedo(on canvas: Canvas) {
        
    }
    
    func didCopyNodes(on canvas: Canvas, nodes: [Node]) {
        
    }
    
    func didPasteNodes(on canvas: Canvas, on layer: CanvasLayer, nodes: [Node]) {
        
    }
    
    func didSelectNodes(on canvas: Canvas, on layer: CanvasLayer, selectedNodes: [Node]) {
        
    }
    
    func didMoveNodes(on canvas: Canvas, movedNodes: [Node]) {
        
    }
    
    
    
    /************************
     *                      *
     *       FUNCTIONS      *
     *                      *
     ************************/
   /*
    @objc func newColor() {
        let colors: [UIColor] = [.green, .blue, .red, .purple, .black]
        let rand = Int(arc4random_uniform(UInt32(colors.count)))
        let nColor = colors[rand]
        canvas.currentBrush.strokeColor = nColor
    }
    
    @objc func newTool() {
        let tools: [CanvasTool] = [.pen, .eraser, .line, .rectangle, .ellipse, .eyedropper, .paint]
        let rand = Int(arc4random_uniform(UInt32(tools.count)))
        canvas.currentTool = tools[rand]
        print("tool: \(canvas.currentTool)")
    }
    
    @objc func selectTool() {
        canvas.currentTool = .selection
    }
    */
    @objc func undo() {
        canvas.undo()
    }
    
    @objc func redo() {
        canvas.redo()
    }
    /*
    @objc func clear() {
        canvas.clear()
    }
    */
    @objc func addLayer() {
        let rand = Int(arc4random_uniform(UInt32(2)))
        let layer = CanvasLayer(type: rand == 0 ? .raster : .vector)
        canvas.addLayer(newLayer: layer, position: .above)
    }
    /*
    @objc func switchLayer() {
        let rand = Int(arc4random_uniform(UInt32(canvas.canvasLayers.count)))
        canvas.switchLayer(to: rand)
    }
    
    @objc func exportImage() {
        let exp = canvas.export()
        UIImageWriteToSavedPhotosAlbum(exp, nil, nil, nil)
    }
    
    
    func alert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: UIAlertController.Style.alert)
        alert.addAction(UIAlertAction(title: "Ok", style: UIAlertAction.Style.cancel, handler: nil))
        self.show(alert, sender: self)
    }
    */
    @objc func zoom(sender: UIPinchGestureRecognizer) {
        let transform = canvas.transform.scaledBy(x: sender.scale, y: sender.scale)
        canvas.transform = transform
        sender.scale = 1.0
    }
    
    
}
