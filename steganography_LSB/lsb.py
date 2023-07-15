from math import inf
import math
import os
import random
import numpy as np
import cv2 as cv


curdir = os.path.dirname(os.path.realpath(__file__))
stru = '\yy'
curdir = curdir.replace(stru[0],'/') + '/'

class CustomException(Exception):
    pass

glb_info_encoded_wr = []
glb_info_encoded_re = []






def custom_dot_mat(a , b):
    res = []
    for ind in range(len(a)):
        row = []
        for inde in range(len(b[0])):
            row.append(0)
        res.append(row)
    for ind in range(len(a)):
        for inde in range(len(b[0])):
            for elind in range(len(b)):
                res[ind][inde] += a[ind][elind] * b[elind][inde]
    return res

def custom_dot_vec(a,b):
    res = []
    for ind in range(len(a)):
        res.append(0)
    for ind in range(len(a)):
        for inde in range(len(a[ind])):
            res[ind] += a[ind][inde] * b[inde]
    return res

#  ITERATIVE METHOD      
def gausseidl(n,mat,b,x,tol,maxIT):
    def ite(v):
        elements = []
        for el in v:
            elements.append(el)
        def calcinw(ind):
            result = b[ind]
            for indx in range(n):
                if(indx!=ind):
                    result = result - mat[ind][indx] * elements[indx]
            return result/mat[ind][ind]

        for ind in range(n):
            elements[ind] = calcinw(ind)
        return elements

    def tole(pre,new):
        mat = np.array(pre) - np.array(new)
        return np.linalg.norm(mat,inf) <= tol
    prev = x
    res = []
    for rn in range(maxIT):
        res = ite(prev)
        if(tole(prev,res)):
            return (list(map(lambda x : round(x) , res)),rn)
        prev = res
    return None
# ________________________________________________________________________________________________________________________________________
# creating K
def create_k(n):
    res = []
    for i in range(n):
        tmp = []
        for j in range(n):
            if( i == j):
                tmp.append((3 * n))
            else :
                tmp.append(random.randint(0,5))
        res.append(tmp)
    return np.array(res)
# _________________________________________________________________________________________________________________________________________

# Gram-Schmid

def gram_schmid_basis(v):
    
    def proj(ve1, ve2):
        return ve1 * (np.dot(ve2, ve1) / np.dot(ve1, ve1))
    u = []
    for i in range(len(v)):
        tmp = v[i]
        lnu = len(u)
        for ind in range(lnu):
            tmp = tmp - proj(u[ind] , v[lnu])
            if(not (np.array(tmp)).any()):
                return None

        u.append(np.array(tmp))
    u = list(map(lambda x : x/np.linalg.norm(x,len(u[0])) , u))
    return u
def check_invertibility(set):
    vecspace = gram_schmid_basis(set)
    if(vecspace is None):
        return False
    return True
# ---------------------------------
def get_y(k,orde):
    return np.dot(k,orde)

def encode_sequence(seq , n):
    stringlength_glb = len(seq)
    encodingks_glb = []
    if(n > stringlength_glb):
        n = stringlength_glb
    if(n > 1000):
        n = 1000
    k = create_k(n)
    while(not check_invertibility(k)):
        k = create_k(n)
    encodingks_glb.append(k)
    result = []
    arr = seq.copy()
    while(len(arr) > 0):
        m = len(arr)
        if(m == 0):
            break
        if(m < n):
            newk = create_k(m)
            while(not check_invertibility(newk)):
                newk = create_k(m)
            encodingks_glb.append(newk)
            for el in get_y(newk,arr):
                result.append(round(el))
            break
        else :
            for el in get_y(k,arr[0:n]):
                result.append(round(el))
        arr = arr[n:]
    return( np.array(result),encodingks_glb)

def encfor_mat(mat, n): 
    x = []
    for el in mat:
        for rgb in el:
            for info in rgb:
                x.append(info)
    return encode_sequence(x,n)


def decode_y(y,encodingks):
    x = []
    arr = y.copy()
    n = len(encodingks[0])
    while(len(arr) > 0):
        m = len(arr)
        if(m < n):
            n = len(arr)
            l = gausseidl(n , encodingks[1],arr,[0] * n , 10**(-10) , 1000)
            for el in l[0]:
                x.append(round(el))
            break
        else :
            l = gausseidl(n , encodingks[0] , arr[0:n],[0] * n, 10**(-10),1000)
            for el in l[0]:
                x.append(el)
        arr = arr[n:]
    return x


# encodingks = []
# imgwidth =0
# imgheight = 0
# stringlength = 0

# istwoks = False
# k1length = 0
# k2length = 0

# first two pixels encoded length of encoded sequence
# last num bit if its for img - 0 , if for string - 1
# second to last bit on last num if there are two k-s = 1 else 0
# second,third, forth and fifth to last num is for first k-length 
# 6,7,8,9 for second k if there exists one
# after k-s have been hidden if we encoded an image we will encode image length on 5nums/10bit

def build_trojanhorse(encodedseq,encodingks,img,isforimg,imglength = 0):
    global glb_info_encoded_wr
    flattenedimg_bin = list(map(lambda x : bin(x)[2:].zfill(8) ,np.array(img.flatten())))
    flatimglen = len(flattenedimg_bin)
    enclength = len(encodedseq)
    if (flatimglen < enclength * 12 + 11 + len(encodingks[0]) * 10 * len(encodingks[0]) + len(encodingks[0]) * 4 ):
        raise CustomException("too much information")
    
    enclength_bin = bin(enclength)[2:].zfill(16)
    kslen = []
    kslen.append(len(encodingks[0]))
    istwoks_glb =len(encodingks)==2
    if(istwoks_glb):
        kslen.append(len(encodingks[1]))

    
    glb_info_encoded_wr = encodedseq

    encoded_bin = list(map(lambda x : bin(x)[2:].zfill(24) ,np.array(encodedseq)))



    # encoding length
    for i in range(8):
        flattenedimg_bin[i] = flattenedimg_bin[i][:-2] + enclength_bin[i * 2] + enclength_bin[(i) * 2 + 1]


    # encoding encoded sequence
    curindex = 8
    previndex = curindex
    for i in range(enclength):
        num = encoded_bin[i]
        for j in range(12):
            flattenedimg_bin[curindex] =  flattenedimg_bin[curindex][:-2] + num[j * 2] + num[j*2+1]
            curindex = curindex + 1


    
    if(isforimg):
        flattenedimg_bin[flatimglen - 1] = flattenedimg_bin[flatimglen - 1][0:-1] + '0'
    else : 
        flattenedimg_bin[flatimglen - 1] = flattenedimg_bin[flatimglen - 1][0:-1] + '1'


    
    firstklen_bin = bin(len(encodingks[0]))[2:].zfill(10)
    if(len(encodingks) == 2):
        flattenedimg_bin[flatimglen - 1] = flattenedimg_bin[flatimglen - 1][0:-2] + '1' + flattenedimg_bin[flatimglen - 1][-1]
        secklen_bin = bin(len(encodingks[1]))[2:].zfill(10)
        for i in range(5):
            a = flatimglen - 2 - i
            flattenedimg_bin[a] = flattenedimg_bin[a][0:-2] + firstklen_bin[i*2] + firstklen_bin[i * 2 + 1]
        for i in range(5):
            a = flatimglen - 7 - i
            flattenedimg_bin[a] = flattenedimg_bin[a][0:-2] + secklen_bin[i*2] + secklen_bin[i * 2 + 1]
        curindex = flatimglen - 12
        kslength = len(encodingks[1])  + len(encodingks[0])
    else :
        flattenedimg_bin[flatimglen - 1] = flattenedimg_bin[flatimglen - 1][0:-2] + '0' + flattenedimg_bin[flatimglen - 1][-1]
        for i in range(5):
            a = flatimglen - 2 - i
            flattenedimg_bin[a] = flattenedimg_bin[a][0:-2] + firstklen_bin[i*2] + firstklen_bin[i * 2 + 1]
        curindex = flatimglen - 7
        kslength = len(encodingks[0])

    
    flat_k_bin = list(map(lambda x : bin(x)[2:].zfill(10) , np.array(encodingks[0]).flatten()))
    if(len(encodingks) == 2):
        flat_k_bin = flat_k_bin  + list(map(lambda x : bin(x)[2:].zfill(10) , np.array(encodingks[1]).flatten()))
    for el in flat_k_bin:
        for ind in range(5):
            flattenedimg_bin[curindex] = flattenedimg_bin[curindex][:-2] + el[ind*2] + el[ind*2 + 1]
            curindex = curindex - 1

    if(isforimg):
        imglength = bin(imglength)[2:].zfill(12)
        for ind in range(6):
            flattenedimg_bin[curindex] = flattenedimg_bin[curindex][0:-2] + imglength[ind*2] + imglength[ind * 2 + 1]
            curindex = curindex - 1
    
    
    flattenedimg_bin = list(map(lambda x : int(x,2),flattenedimg_bin))

    imgheight = len(img)
    imagewidth = len(img[0])
    newimg = []
    for i in range(imgheight):
        row = []
        for j in range(imagewidth):
            a = (i * imagewidth + j) * 3
            row.append([flattenedimg_bin[a] , flattenedimg_bin[a + 1],flattenedimg_bin[a + 2]])
        newimg.append(row)

    return np.array(newimg).astype(np.uint8)


# first two pixels encoded length of encoded sequence
# last num bit if its for img - 0 , if for string - 1
# second to last bit on last num if there are two k-s = 1 else 0
# second,third, forth and fifth to last num is for first k-length 
# 6,7,8,9 for second k if there exists one
# each k weight is encoded in 5 nums, -10bits
# after k-s have been hidden if we encoded an image we will encode image length on 5nums/10bit

def decipherimage(img):
    global glb_info_encoded_re
    flattened_bin = list(map( lambda x : bin(x)[2:].zfill(8),np.array(img).flatten()))
    seq_length = ''
    for i in range(8):
        seq_length =  seq_length + flattened_bin[i][-2:]
    seq_length = int(seq_length,2)

    encoded_bin = []
    curindex = 8
    for i in range(seq_length):
        el = ''
        for j in range(12):
            el = el + flattened_bin[curindex][-2:]
            curindex = curindex + 1
        encoded_bin.append(el)

    encoded_bin = np.array(list(map(lambda x : int(x,2) , encoded_bin)))
    
    curindex = len(flattened_bin) - 1
    isimg = (flattened_bin[curindex][-1] == '0')
    twoks = flattened_bin[curindex][-2] == '1'
    curindex = curindex - 1
    firstklen = ''
    secklen = ''
    for i in range(5):
        firstklen = firstklen + flattened_bin[curindex][-2:]
        curindex = curindex - 1
    firstklen = int(firstklen,2)
    if(twoks):
        for i in range(5):
            secklen = secklen + flattened_bin[curindex][-2:]
            curindex = curindex - 1
        secklen = int(secklen,2)

        

    k1_bin = []
    for _ in range(firstklen * firstklen):
        el = ''
        for _ in range(5):
            el = el + flattened_bin[curindex][-2:]
            curindex = curindex - 1
        k1_bin.append(el)
    
    k1 = list(map(lambda x : int(x,2) , k1_bin))
    k1_mat = []
    for i in range(firstklen):
        a = i * firstklen
        k1_mat.append(k1[a :a + firstklen])

    encodingks = [k1_mat]
    if(twoks):
        k2_bin = []
        for _ in range(secklen * secklen):
            el = ''
            for _ in range(5):
                el = el + flattened_bin[curindex][-2:]
                curindex = curindex - 1
            k2_bin.append(el)
    
        k2 = list(map(lambda x : int(x,2) , k2_bin))
        k2_mat = []
        for i in range(secklen):
            a = i * secklen
            k2_mat.append(k2[a :a + secklen])
        encodingks.append(k2_mat)

    if(isimg):
        imglength = ''
        for i in range(6):
            imglength = imglength + flattened_bin[curindex][-2:]
            curindex = curindex - 1
        imglength = int(imglength,2)
    


    glb_info_encoded_re = encoded_bin

    sequence = decode_y(encoded_bin,encodingks)
    if(isimg):
        result = []
        curindex = 0
        imgheight = int(len(sequence) / (imglength * 3))
        for i in range(imgheight):
            row = []
            for j in range(imglength):
                row.append([sequence[curindex] , sequence[curindex + 1] , sequence[curindex + 2]])
                curindex = curindex + 3
            result.append(row)
        result = np.array(result).astype(np.uint8)
    else:
        result = ''
        for el in sequence:
            if(el <= 0):
                el = -el
            result = result + chr(el)
    return result
                

    
    

    
    
def create_spy_image_text(image_name, newname, info, n = 20, dir = curdir , newdir = curdir):
    if(np.array(info).ndim > 1):
        raise CustomException('this need to be a text')
    matrix = cv.imread(dir + image_name)
    enc_info  =  list(map(lambda x : ord(x), info))
    enc_info = encode_sequence(enc_info, n)
    matrix = build_trojanhorse(enc_info[0],enc_info[1],matrix,False)
    cv.imwrite(newdir + newname , matrix)

def create_spy_image_img(host_image_name, newname, parasite_image_name,  n = 50, host_dir = curdir , parasite_dir = curdir, newdir = curdir):
    info = cv.imread(parasite_dir + parasite_image_name)
    if(np.array(info).ndim <= 1):
        raise CustomException('i dont think this is an image')
    matrix = cv.imread(host_dir + host_image_name)
    enc_info  =  encfor_mat(info,n)
    matrix = np.array(build_trojanhorse(enc_info[0],enc_info[1],matrix,True,len(info[0]))).astype(np.uint8)
    cv.imwrite(newdir + newname , matrix)


def check_image(image_name , dir = curdir):
    matrix = np.array(cv.imread(dir + image_name))
    return decipherimage(matrix)

create_spy_image_text('arthas.jpg' , '1m.png' , 'ილია ჭავჭავაძე (დ. 8 ნოემბერი, 1837, სოფელი ყვარელი — გ. 12 სექტემბერი, 1907, წიწამური) — ქართველი საზოგადო მოღვაწე, პუბლიცისტი, ჟურნალისტი, პოლიტიკოსი, მწერალი, რუსეთის იმპერიის სახელმწიფო საბჭოს დეპუტატი 1906-1907 წლებში, მნიშვნელოვანი როლი შეასრულა მეცხრამეტე საუკუნეში საქართველოში სამოქალაქო საზოგადოების ჩამოყალიბებაში, გადამწყვეტი წვლილი შეიტანა საქართველოს ეროვნულ-განმათავისუფლებელი მოძრაობის შექმნაში და ლიბერალური ფასეულობების გავრცელებაში.სათავეში ჩაუდგა თერგდალეულთა თაობას, რომლებმაც ქართულ ინტელექტუალურ სივრცეში მოდერნული, ევროპული იდეები და ხედვები შემოიტანეს. ილია ჭავჭავაძის თაოსნობით დაარსდა და სრულიად ახალი სიტყვა თქვა ქართულ ჟურნალისტიკაში მის მიერ გამოცემულმა გაზეთებმა „საქართველოს მოამბემ“ და „ივერიამ“. მნიშვნელოვანი წვლილი შეიტანა საქართველოში პირველი ფინანსური დაწესებულების - სათავადაზნაურო-საადგილმამულო ბანკის შექმნაში, რომელსაც 30 წლის განმავლობაში ხელმძღვანელობდა. ბანკის რესურსი უმეტესწილად საქართველოში სხვადასხვა კულტურულ, საგანმანათლებლო, ეკონომიკურ, საქველმოქმედო პროექტებს ხმარდებოდა. მანვე მნიშვნელოვანი როლი ითამაშა ქართველთა შორის წერა-კითხვის გამავრცელებელი საზოგადოების ჩამოყალიბებასა და ფუნქციონირებაში. მისი თაოსნობით გაიხსნა არაერთი სკოლა, სადაც სწავლება ქართულ ენაზე მიმდინარეობდა, რამაც საქართველოში რუსიფიკაციის პროცესი შეაჩერა.ილია ჭავჭავაძის პუბლიცისტიკას ქართული ეროვნული და სამოქალაქო ცნობიერების ფორმირებაში უდიდესი როლი მიუძღვის. მისი წერილების თემატიკა მრავალფეროვანია და მოიცავს: ეროვნულ საკითხებს, ლიტერატურას, განათლებას, თეატრს, ეკონომიკას, სოციალურ-პოლიტიკურ და მსოფლიოში მიმდინარე პროცესებს. მის პუბლიცისტიკაში ასახულია ევროპული იდეები, ხედვები, რომელთა გაცნობაც ჟურნალ-გაზეთების საშუალებით მთელი ქართული საზოგადოებისთვის ხდებოდა. მნიშვნელოვანია მისი შეხედულებები თვითმმართველობის, სასამართლო სისტემის, ეკონომიკის, სოციალური წესრიგის, ადამიანის უფლებების, ქალთა უფლებების, სასკოლო და უმაღლესი განათლების, სამოქალაქო აქტივიზმის შესახებ. აღსანიშნავია ილია ჭავჭავაძის წერილები საგარეო საკითხებზე, რამაც ქართული საზოგადოება მეტად დააახლოვა დასავლეთსა და ზოგადად მსოფლიოში მიმდინარე პროცესებთან. აღსანიშნია, რომ ევროპელობის, საქართველოს ევროპული მომავლის ნარატივი ყველაზე მკაფიოდ პირველად ილია ჭავჭავაძისა და სხვა თერგდალეულთა ნააზრევში გამოჩნდა.რუსეთში მიმდინარე რევოლუციის ფონზე, ილია ჭავჭავაძე აირჩიეს ქართველ თავადაზნაურთა წარმომადგენლად რუსეთის იმპერიის სახელმწიფო საბჭოში. მან სცადა გამოეყენებინა ეს ტრიბუნა არა ერთი რომელიმე წოდების, არამედ მთლიანად ქართველი ხალხის ინტერესების დასაცავად. მონაწილეობდა სიკვდილით დასჯის შესახებ დებატებში, ასევე ლობირებდა საქართველოსთვის ავტონომიის მოპოვების საკითხს. ილია ჭავჭავაძის მხატვრული შემოქმედება ეროვნულ და სოციალურ მოტივებზეა აგებული. ნაციონალურ საკითხთან მიმართებაში მისი ნაწერები, როგორებიცაა აჩრდილი, ქართვლის დედას და დიმიტრი თავდადებული სამშობლოსადმი თავგანწირვასა და მის უპირობო სიყვარულს ქადაგებს. სოციალურ საკითხებთან მიმართებაში ილია ჭავჭავაძე თავისი ნაწარმოებებით, როგორებიცაა რამდენიმე სურათი ანუ ეპიზოდი ყაჩაღის ცხოვრებიდან, გლახის ნაამბობი, კაცია-ადამიანი?! და ოთარაანთ ქვრივი, ილაშქრებს სოციალური უთანასწორობის და ფეოდალური ღირებულებების წინააღმდეგ')
newbird =  check_image('1m.png')
print(newbird)
