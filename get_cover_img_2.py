# coding:utf-8
import numpy as np
import cv2 


#############################################################
# 機能　書籍を自動認識して撮影する、
# 入口　get_cover_img()
# 操作　自動撮影できない場合は、スペースキーで手動撮影する
# return 0:自動撮影成功
#        1:手動撮影成功
#        2:撮影中止
# 出力　カレントディレクトリに下記画像を保存する
#　　　　result_image.jpg　　撮影した画像
#　　　　result_trimming.jpg　上記画像をトリミングした画像
#　　　　result_trimming_modified.jpg 上記画像の歪みを修正した画像
#############################################################
# 注意　複数カメラを接続している場合、50行を修正して使用
#############################################################
# 使用　0 を受信したら、result_trimming_modified.jpg　を結果とする
# 　　　1 を受信したら、result_image.jpg　を結果とする
#############################################################

# 歪み補正
def modify_image(image,approx):

    height, width, _ = image.shape
    approx=approx.tolist()

    left = sorted(approx,key=lambda x:x[0]) [:2]
    right = sorted(approx,key=lambda x:x[0]) [2:]

    left_down= sorted(left,key=lambda x:x[0][1]) [0]
    left_up= sorted(left,key=lambda x:x[0][1]) [1]

    right_down= sorted(right,key=lambda x:x[0][1]) [0]
    right_up= sorted(right,key=lambda x:x[0][1]) [1]

    perspective1 = np.float32([left_down,right_down,right_up,left_up])
    perspective2 = np.float32([[0, 0],[width, 0],[width, height],[0, height]])  

    psp_matrix = cv2.getPerspectiveTransform(perspective1,perspective2)
    img_psp = cv2.warpPerspective(image, psp_matrix,(width,height))

    return img_psp

# 入口
def get_cover_img():
    color = (  0,  0,255)
    capture = cv2.VideoCapture(0) # カメラ番号を選択、例えば　capture = cv2.VideoCapture(1)
    if capture.isOpened() is False:
        raise("IO Error")
    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
    while True:
        _, image = capture.read()
        #image_mirror = image[:,::-1]

        src = image
        cv2.imshow("Capture", src )
        height, width, _ = src.shape
        image_size = height * width
        # グレースケール化
        img_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        mean = cv2.mean(img_gray)

        # しきい値指定によるフィルタリング
        #retval, dst = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV )
        _, dst = cv2.threshold(img_gray, int(mean[0]) , 255, cv2.THRESH_TOZERO_INV )
        # 白黒の反転
        dst = cv2.bitwise_not(dst)
        # 再度フィルタリング
        _, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 輪郭を抽出
        dst, contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)

        for i, contour in enumerate(contours):
            # 小さな領域の場合は間引く
            dst1 = src
            area = cv2.contourArea(contour)
            if area < image_size * 0.70:
                print('area < image_size * 0.70')
                continue
            # 画像全体を占める領域は除外する
            if image_size * 0.99 < area:
                continue
            
            # 外接矩形を取得

            arclen = cv2.arcLength(contour,
                               True) # 対象領域が閉曲線の場合、True
            approx = cv2.approxPolyDP(contour,
                                  0.05*arclen,  # 近似の具合?
                                  True)
           
            if len(approx) == 4:
                image_modified = modify_image(image,approx)
                cv2.imwrite("result_trimming_modified.jpg",image_modified) # トリミングした画像の歪みを修正した画像


                x,y,w,h = cv2.boundingRect(contour)
                wid = (x+w if  x+w < width else width -1)
                hei = (y+h if y+h < height else height -1)
                img = image[y:hei,x:wid ]
                
                # print('x=',x,' y=',y,' w=',w,' h=',h)
                # print('height =',height,' width=',width)
                # print('len(approx)=',len(approx))

                dst1 = cv2.drawContours(image,
                            [approx],
                            -1,    # 表示する輪郭. 全表示は-1
                            color,
                            2)    # 等高線の太さ
                cv2.imwrite('result_trimming.jpg', img) # トリミングした画像
                cv2.imwrite('result_image.jpg', image)  #　自動撮影した画像
            
                return (0)
        cv2.imshow("Capture", image )

        keyInput = cv2.waitKey(3) # 撮影速度 大きくなると遅くなる
        if keyInput == 32: # when Space
            cv2.imwrite('result_image.jpg', image)  #　手動撮影した画像
            return (1)
        if keyInput == 27: # when ESC
            return (2)  #　撮影中止
    
if __name__ == '__main__':
    r = get_cover_img()
    if r == 0:
        print('自動撮影成功')
    if r == 1:
        print('手動撮影成功')
    if r == 2:
        print('撮影中止された')


