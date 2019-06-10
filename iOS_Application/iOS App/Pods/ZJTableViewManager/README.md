
[English introduction](https://github.com/JavenZ/ZJTableViewManager/blob/master/README_EN.md)
### 关于ZJTableViewManager
最近开始用Swift写项目，在这之前只看了看Swift相关的文档，突然开始写很不适应，特别是之前一直在用的数据驱动的TableView框架`RETableViewManger`没有Swift版，混编的话也有问题（可能是Swift 4.0不兼容）于是就决定自己写一下Swift版的，使用方式基本一致。但是加了一些扩展功能，比如cell高度的自动计算等等。

### Swift版本适配
Swift 4.0/4.2




举个例子，要实现下面这个界面只需要10行代码：

![image](https://github.com/JavenZ/ZJTableViewManager/blob/master/ScreenShot/forms_shot.jpg)

使用TableView初始化ZJTableViewManager，添加一个section，section里面添加cell
```swift
        self.manager = ZJTableViewManager(tableView: self.tableView)
        
        //add section
        let section = ZJTableViewSection()
        self.manager.add(section: section)
        
        //Simple String
        section.add(item: ZJTableViewItem(title: "Simple String"))
        
        //Full length text field
        section.add(item: ZJTextFieldItem(title: nil, placeHolder: "Full length text field", text: nil, isFullLength: true, didChanged: nil))
        
        //Text Item
        section.add(item: ZJTextFieldItem(title: "Text Item", placeHolder: "Text", text: nil,  didChanged: nil))
        
        //Password
        let passwordItem = ZJTextFieldItem(title: "Password", placeHolder: "Password Item", text: nil,  didChanged: nil))
        passwordItem.isSecureTextEntry = true
        section.add(item: passwordItem)
        
        //Switch Item
        section.add(item: ZJSwitchItem(title: "Switch Item", isOn: false,  didChanged: nil))

```
到这里，这个界面就搭建好了，add item的顺序就是界面上cell的展示顺序，不需要写tableview的代理。didChanged是界面上text变化或者按钮触发的回调，可以实时获取。

### 关于数据驱动
使用TableView时，复杂TableView界面cell数量多的时候，VC里面的每个delegate、dataSource方法都要写if else判断，特别是`tableView(_:cellForRowAt:)`里面代码会很多，即使想办法封装，里面大量的if else也避免不了。
数据驱动搭建TableView页面，就是为了处理这个情况而诞生的。开发者不需要处理TableView的delegate、dataSource，只需要关心数据的处理。数据处理好并赋值，页面就按照数据的样子搭建起来。
同时对TableView界面的动态改变也不需要通过TableView或者cell，而是直接对cell对应的item做操作，数据的处理和TableView的显示效果关联，逻辑处理更加集中，代码更加简洁，更加容易实现复杂的TableView界面，同时代码也更加容易理解。

### 界面操作
**都可以使用系统自带的动画**

代码删除cell：
```
passwordItem.delete(.automatic)
```

侧滑删除
```swift
item.editingStyle = .delete
item.setDeletionHandler(deletionHandler: {[weak self] (item) in
      self?.deleteConfirm(item: item)
      })
```

刷新cell：
```
passwordItem.reload(.automatic)
```

更新cell的高度：
```
item.cellHeight = 200
//这个方法只更新cell高度，自带动画，不会reload这个cell
item.updateHeight()
```
批量在某个section里面插入 cell：
```swift
section.insert(arrItems, afterItem: item, animate: .fade)
```
批量删除cell：
```swift
section.delete(arrItems, animate: .fade)
```


刷新section：
```
section.remove(item: simpleStringItem)
section.remove(item: passwordItem)
section.reload(.automatic)
```

sectionHeader 在屏幕上出现、消失的回调
```swift
section.setHeaderWillDisplayHandler({ (currentSection) in
                print("Section" + String(currentSection.index) + " will display!")
            })
            
section.setHeaderDidEndDisplayHandler({ (currentSection) in
                print("Section" + String(currentSection.index) + " did end display!")
            })
```

### 自动计算高度：
ZJTableViewManager提供了提前计算cell高度并缓存的api，只需要调用一行代码即可实现自动计算高度，高度和item绑定，不需要缓存，效率很高。
```swift
item.autoHeight(manager)
```
要提前把item的相关属性赋值好，再调用`item.autoHeight(manager)`，具体可查看demo



### Demo：
电商项目的评价、打星评分、添加评论图片,

![image](https://github.com/JavenZ/ZJTableViewManager/blob/master/ScreenShot/pictureitem_edit.gif?raw=true)    ![image](https://github.com/JavenZ/ZJTableViewManager/blob/master/ScreenShot/pictrue_item_read.gif?raw=true)

这里主要有3个cell，一个打星的cell，一个评论的cell，一个添加图片的cell。viewController里只有20行代码，耦合性低。

```swift
override func viewDidLoad() {
        super.viewDidLoad()
        self.title = "Demo"
        self.manager = ZJTableViewManager(tableView: self.tableView)
        
        //register cell
        self.manager?.register(OrderEvaluateCell.self, OrderEvaluateItem.self)
        self.manager?.register(ZJPictureTableCell.self, ZJPictureTableItem.self)
        
        //add section
        let section = ZJTableViewSection(headerHeight: 10, color: UIColor.init(white: 0.9, alpha: 1))
        self.manager?.add(section: section)
        
        //add cells
        for i in 0...10 {
          i  //评价cell
            section.add(item: OrderEvaluateItem(title: "评价"))
            let textItem = ZJTextItem(text: nil, placeHolder: "请在此输入您的评价~", ddChanged: nil)
            textItem.isHideSeparator = true
            section.add(item: textItem)
            
            //图片cell
            if i%2 == 1 {
                //只展示图片
                let pictureItem = ZJPictureTableItem(maxNumber: 5, column: 4, space: 15, width: self.view.frame.size.width, superVC: self, pictures: [image])
                pictureItem.type = .read
                section.add(item: pictureItem)
            }else{
                //添加图片
                let pictureItem = ZJPictureTableItem(maxNumber: 5, column: 4, space: 15, width: self.view.frame.size.width, superVC: self)
                pictureItem.type = .edit
                section.add(item: pictureItem)
            }
        }
        
        // Do any additional setup after loading the view, typically from a nib.
    }
```

### 使用
直接拖入ZJTableViewManager文件夹里面的文件，或者用cocoapods
`pod 'ZJTableViewManager'`

### 注：
1.tableView可以storyboard、xib、纯代码初始化

2.我自己使用主要是用xib方式搭建cell，但纯代码的cell也支持




