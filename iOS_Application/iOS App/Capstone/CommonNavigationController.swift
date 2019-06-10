//
//  commonNavigationViewController.swift
//  Capstone
//
//  Created by Kim Chan Il on 30/04/2019.
//  Copyright © 2019 Kim Chan Il. All rights reserved.
//


import Foundation
import UIKit


class CommonNavigationController: UINavigationController, UINavigationControllerDelegate, UINavigationBarDelegate {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.delegate = self
    }
    
    func navigationController(_ navigationController: UINavigationController, willShow viewController: UIViewController, animated: Bool) {
        //if viewController.restorationIdentifier == "addVC" || viewController.restorationIdentifier == "categoryVC" || viewController.restorationIdentifier == "repeatVC" || viewController.restorationIdentifier == "categoryEdit" || viewController.restorationIdentifier == "setSleep" || viewController.restorationIdentifier == "login" || viewController.restorationIdentifier == "scheduleView" {
      //      navigationController.navigationBar.isHidden = false
        //}
        //else{
        navigationController.navigationBar.isHidden = true
        //}
    }
    
    
    @objc func tapBack() {
        self.navigationController?.popViewController(animated: true)    }
    
}
