//
//  settingsViewController.swift
//  Capstone
//
//  Created by Kim Chan Il on 30/04/2019.
//  Copyright © 2019 Kim Chan Il. All rights reserved.
//

import UIKit
import ZJTableViewManager
import RETableViewManager
class settingsViewController: UIViewController {
    
    var tableview:UITableView!
    var manager:ZJTableViewManager!
    
    
    var Section1:ZJTableViewSection = {
        var section = ZJTableViewSection()
        var cell = RETableViewCell()
        
        cell.layer.masksToBounds = true
        //cell.layer.cornerRadius = 20
        cell.layer.maskedCorners = [.layerMinXMinYCorner, .layerMaxXMinYCorner]
        cell.textLabel!.font = UIFont.boldSystemFont(ofSize: 40)
        cell.textLabel!.text = "설정"
        cell.textLabel!.textColor = UIColor(red: 250/255, green: 128/255, blue: 114/255, alpha: 1)
        cell.backgroundColor = UIColor.white
        cell.isUserInteractionEnabled = false
        cell.separatorInset.left = 15
        
        section.headerView = cell
        section.headerHeight = 80
        
        section.footerHeight = 0
        return section
    }()
    
    var Section2:ZJTableViewSection = {
        var section = ZJTableViewSection()
        return section
    }()
    
    var homepage:ZJTableViewItem = {
        var item = ZJTableViewItem(title: " Homepage")
        
        return item
    }()
    var contactus:ZJTableViewItem = {
        var item = ZJTableViewItem(title: " ContactUs")
        
        return item
    }()
    var appVer:ZJTableViewItem = {
        var item = ZJTableViewItem(title: " Version : 1.0.0")
        
        
        return item
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        tableview = UITableView(frame: CGRect(x: 0, y: 0, width: self.view.frame.width, height: self.view.frame.height))
        tableview.separatorStyle = .none
        
        manager = ZJTableViewManager(tableView: tableview)
        
        Section1.add(item: homepage)
        Section1.add(item: contactus)
        Section2.add(item: appVer)
        manager.add(section: Section1)
        manager.add(section: Section2)
        
        self.view.addSubview(tableview)
        
    }
    
}
