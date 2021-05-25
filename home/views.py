from django.shortcuts import render, HttpResponse
import time
# import final_func
import os
from django.conf import settings

import tweepy
import math
import datetime
import pandas as pd
# import time
import numpy as np
import urllib
from urllib.request import urlopen
import tensorflow as tf
import cv2
from joblib import dump, load
from skimage import io
from tensorflow.keras.models import load_model


def establish_connection():
    global auth
    global api
    auth = tweepy.OAuthHandler("Ueyphk6rJS5wFjnFKYxPLuhIr", "9bVtiSWrXOBFPjDwCXZynJ0W8uYpd8KMhsJcRyG63ocD3hzkMM")
    auth.set_access_token('1319706669245292544-wkRstlNWriivHWdxTX7EXkV60JvH9R', 'KibZ5HWNx82bNQ5nyaXNADarY08mzr4t1Vdz3evdHo9Og')
    api = tweepy.API(auth)

def user_active(screen_name):
  flag = True
  user = 0
  try:
    user = api.get_user(screen_name) 
    followers_count = user.followers_count 
  except tweepy.TweepError as e:
        if str(e).find("Rate limit exceeded") != -1:
            print("Rate limit Error: Execution stopped for 15*61 seconds.")
            print(f"Program paused at    : {datetime.datetime.now()+ datetime.timedelta(minutes=5.5*60)} Indian Standard Time (IST)")
            print(f"Expected resume time : {datetime.datetime.now() + datetime.timedelta(seconds=15*61) + datetime.timedelta(minutes=5.5*60)} Indian Standard Time (IST)")
            print("Program at user_active(screen_name)")
            print(f"params => {screen_name}")
            time.sleep(15 * 61)
            try:
              user = api.get_user(screen_name) 
              followers_count = user.followers_count 
              
            except:
              if str(e).find("User has been suspended") != -1 or str(e).find("User not found") != -1:
                print(f'User "{screen_name}" returned user_suspened/user_not_found error. Action : skipped')  
                flag = False
                
              else:
                flag = False
                print(f'User "{screen_name}" returned error :{str(e)}. Action : skipped')  
                

        elif str(e).find("User has been suspended") != -1 or str(e).find("User not found") != -1:
              flag = False
              print(f'User "{screen_name}" returned user_suspened/user_not_found error. Action : skipped')  
                
        else:
            flag = False
            print(f'User "{screen_name}" returned error :{str(e)}. Action : skipped')  
                
  return [flag,user]

############################################
### Call function only if User is active ###
############################################


def user_timeline_fetch(screen_name, count_limit):
    
    # params      ===> screen_name  : accounts screen name without @
    # count_limit ===> max number of tweets to be fecthed (not exact)
    tweets_list = []
    count = 0
    for i in range(1,100):

        if count > count_limit:
            break
        try:
            time_line = api.user_timeline(screen_name, page = i)
        
        except tweepy.TweepError as e:
            if str(e).find("Rate limit exceeded") != -1:
                print("Rate limit Error: Execution stopped for 15*61 seconds.")
                print(f"Program paused at    : {datetime.datetime.now()+ datetime.timedelta(minutes=5.5*60)} Indian Standard Time (IST)")
                print(f"Expected resume time : {datetime.datetime.now() + datetime.timedelta(seconds=15*61) + datetime.timedelta(minutes=5.5*60)} Indian Standard Time (IST)")
                print("Program at user_timeline_fetch(screen_name)")
                print(f"params => {screen_name}")
                time.sleep(15 * 61)
                try:
                    time_line = api.user_timeline(screen_name, page = i) 
                
                except:
                    print(f'User "{screen_name}"\'s timeline returned error :{str(e)}. Action : skipped')
                    return [0]  
                    
            
            else:
                print(f'User "{screen_name}"\'s timeline returned error :{str(e)}. Action : skipped')  
                return [0]    





        if len(time_line) == 0:
            break
        for tweet in time_line:
            tweets_list.append(tweet)
            count = count + 1
            print(f"Access Tweet No. : {count}")
        
    return tweets_list

###########################################################################################
######## Function returns list as follow:: ################################################
## [retweets, original_tweets, tweet-retweet ratio,, overall_freq_percentage_for_week,
##  overall_freq_percentage_for_week, overall_freq_percentage_for_week, overall_freq_for_week, retweet_freq_for_week, original_tweet_freq_for_week]
#######################################################################################


def retweet_perc(tweets_list):
    retweet = 0
    ori_tweet = 0
    retweet_list = []
    ori_list = [] 
    overall_perc = [0, 0, 0, 0, 0, 0, 0]
    retweet_perc = [0, 0, 0, 0, 0, 0, 0]
    ori_perc = [0, 0, 0, 0, 0, 0, 0]
    
    for t in tweets_list:
        p = t.text.find('RT @')
        if p != -1:
            retweet = retweet + 1
            retweet_list.append(t)
        else:
            ori_list.append(t)
            ori_tweet = ori_tweet + 1
    
    retweet_distribution = week_distribution(retweet_list)
    ori_distribution = week_distribution(ori_list)
    overall_distribution = [x + y for x, y in zip(retweet_distribution, ori_distribution)]

    if len(tweets_list) != 0:
        overall_perc = [x / len(tweets_list) for x in overall_distribution]
    if len(retweet_list) != 0:
        retweet_perc = [x / len(retweet_list) for x in retweet_distribution]
    if len(ori_list) != 0:
        ori_perc = [x / len(ori_list) for x in ori_distribution]
    re_ori_ratio = retweet
    if ori_tweet!=0:
        re_ori_ratio /= ori_tweet
    sorted_overall = overall_perc
    sorted_overall.sort()
    return [retweet, ori_tweet,re_ori_ratio] + overall_perc + retweet_perc + ori_perc + overall_distribution + retweet_distribution + ori_distribution + sorted_overall

def week_distribution(tweets_list):
    # tweets_list : list returned by user_timeline_fetch() 
    week_freq = [0, 0, 0, 0, 0, 0, 0]
    for t in tweets_list:
        x = t.created_at
        week_freq[x.weekday()] = week_freq[x.weekday()] + 1
    return week_freq

def face_flag(user):
    url = user.profile_image_url
    url = url.replace('_normal', '')
    response = urllib.request.urlopen(url)
    image = face_recognition.api.load_image_file(response)
    res = face_recognition.api.face_encodings(image)
    if len(res) == 0:
        return 0
    else: 
        return 1

def files_include():
    global face_cascade
    global face_cascade_side
    global body_cascade
    global ann
    global sc
    face_cascade = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR,'home/neural_asset/haarcascade_frontalface_default.xml'))
    face_cascade_side = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR,'home/neural_asset/haarcascade_profile.xml'))
    body_cascade = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR,'home/neural_asset/haarcascade_fullbody.xml'))
    ann = tf.keras.models.load_model(os.path.join(settings.BASE_DIR,'home/neural_asset/final_model/final_model_loc.h5'))
    sc = load(os.path.join(settings.BASE_DIR,'home/neural_asset/final_model/scaler_loc.joblib'))


def face_dec(user):
    url = user.profile_image_url
    url = url.replace('_normal', '')
    image = io.imread(url) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    face1 = face_cascade.detectMultiScale(gray, 1.3, 5)
    face2 = face_cascade_side.detectMultiScale(gray, 1.3, 5)
    face3 = face_cascade_side.detectMultiScale(cv2.flip(gray, 1), 1.3, 5)
    body  = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(face1)+len(face2)+len(face3) + len(body) == 0:
        return 0
    else: 
        return 1

#######################################################################################################################
######### Function returns list as [no_of_clusters,no_of_clusters_per_day,no_of_tweets,no_of_tweets_per_day] ##########
#######################################################################################################################
#57 done

def timestamp_custom_clustering(tweet_list):
    if len(tweet_list)<3:
        return [0,0,0,0,0]
    sec_time = []
    for x in tweet_list:
        sec_time.append((x.created_at.timestamp()/60))

    minx = min(sec_time)
    sec_time = [x - minx for x in sec_time]
    sec_time.sort()
    cluster_count = 0
    cluster_details = []
    flag = True
    for i in range(len(sec_time)):
        # print(i)
        if flag:
            start = i
            end = i
            flag = False
        elif sec_time[i]-sec_time[i-1] <= 1.5:
            end += 1
        elif flag == False:
            cluster_details.append((start,end))
            start = i
            end = i

        else:
            pass
            #situation never occurs
    cluster_details.append((start,end))
    no_of_clusters = len(cluster_details)
    no_of_tweets = len(sec_time)
    cluster_ratio = no_of_clusters/no_of_tweets
    nod = sec_time[no_of_tweets-1]/(60*24)
    no_of_clusters_per_day = no_of_clusters/nod
    no_of_tweets_per_day = no_of_tweets/nod
    no_of_tweets_per_cluster = no_of_tweets/no_of_clusters

    return [no_of_clusters,no_of_clusters_per_day,no_of_tweets_per_cluster,no_of_tweets,no_of_tweets_per_day]

######### Returns total no of apperances keyword 'follow' ,total no of tweets, and apperances per tweet
# 63 done
def keyword_analysis(tweet_list):
    count = 0;
    total = 0;
    for tweet in tweet_list:
        count += tweet.text.count('follow')
        count += tweet.text.count('Follow')
        count += tweet.text.count('like')
        count += tweet.text.count('Like')
        count += tweet.text.count('tweet')
        count += tweet.text.count('Tweet')
        total += 1
    if total == 0:
        return[0,0]
    return [count, count/total]

def fetch_user_details(screen_name):
    user = user_active(screen_name)
    if not user[0]:
        return []
    tweet_list = user_timeline_fetch(screen_name, 1000)
    if len(tweet_list) == 1 and tweet_list[0] == 0:
        return [] 
    retweet_analysis = retweet_perc(tweet_list)
    cluster_analysis = timestamp_custom_clustering(tweet_list)
    keyword_result = keyword_analysis(tweet_list)
    no_followers = user[1].followers_count
    no_following = user[1].friends_count
    description_present = 0
    if user[1].description != '':
        description_present = 1
    profile_image = 0
    if user[1].default_profile_image == True:
        profile_image = 1
    follow_following_ratio = no_followers
    if no_following != 0:
        follow_following_ratio = no_followers/no_following
    face = 0
    if profile_image == 0:
        face = face_dec(user[1])


    # print(len(cluster_analysis))
    return [no_followers,no_following,follow_following_ratio,description_present,profile_image] + retweet_analysis + cluster_analysis + [face] + keyword_result

def Average(lst):
    if(len(lst)==0):
        return 0
    return sum(lst) / len(lst)
def user_genu(res):
    count = 0
    for x in res:
        if(x >= 0.7):
            count =count+1
    return count
def user_collu(res):
    count = 0
    for x in res:
        if(x <= 0.4):
            count =count+1
    return count

def user_unsure(res):
    count = 0
    for x in res:
        if(x < 0.7 and x>0.4):
            count =count+1
    return count
def graph_count(x,y,res):
    count = 0
    for p in res:
        if(p>x and p<=y):
            count =count+1
    return count
def listjoin(res):
    op = ''
    for x in res:
        op = op + str(x) + "_list_sep_esc_"
    return op





# len(user)

def complete_user_analysis(screen_name):
    establish_connection()
    files_include()
    output = ""
    usss = user_active(screen_name)
    if(usss[0] == False):
        return output
    followers = tweepy.Cursor(api.followers, screen_name).items(10)  
    # printing the latest 20 followers of the user
    res = []
    sn =[]
    dp=[]
    un=[]
    for follower in followers:
        print(f"User : {follower.screen_name}")
        user = fetch_user_details(follower.screen_name)
            
        try:
            x1 = ann.predict(sc.transform([user]))
        except:
            continue

        res = res + [float(format(x1[0][0], '.4f'))]
        sn = sn + [follower.screen_name]
        dp = dp +[follower.profile_image_url.replace('_normal', '')]
        un = un + [follower.name]
        # print(f"{follower.screen_name} : {x1[0][0]}")

    user = fetch_user_details(screen_name)

    x1 = ann.predict(sc.transform([user]))
    print("Prepare to respond")
    user_acc = user_active(screen_name)
    user_acc = user_acc[1]
    user_name = user_acc.name
    follwer_count = user_acc.followers_count 
    following_count = user_acc.friends_count
    dp_url = user_acc.profile_image_url.replace('_normal', '')
    week0 = format(user[8]*100, '.2f')
    week1 = format(user[9]*100, '.2f')
    week2 = format(user[10]*100, '.2f')
    week3 = format(user[11]*100, '.2f')
    week4 = format(user[12]*100, '.2f')
    week5 = format(user[13]*100, '.2f')
    week6 = format(user[14]*100, '.2f')
    created_at = user_acc.created_at
    desc = user_acc.description
    per_index = format(x1[0][0], '.3f')
    col_index = format(Average(res), '.3f')
    retweet_perc = user[7]
    retweet_perc = format(100*retweet_perc/(retweet_perc +1), '.2f')
    noTweet = user_acc.statuses_count
    keyPTweet = format(user[64], '.4f')
    clusters = user[57]
    TPcluster = format(user[59], '.4f')
    noTweetPDay = format(user[61], '.4f')
    genu = user_genu(res)
    collusive = user_collu(res)
    unsure = user_unsure(res)
    g0 = graph_count(-1,0.05,res)
    g1 = graph_count(0.05,0.15,res)
    g2 = graph_count(0.15,0.25,res)
    g3 = graph_count(0.25,0.35,res)
    g4 = graph_count(0.35,0.45,res)
    g5 = graph_count(0.45,0.55,res)
    g6 = graph_count(0.55,0.65,res)
    g7 = graph_count(0.65,0.75,res)
    g8 = graph_count(0.75,0.85,res)
    g9 = graph_count(0.85,0.95,res)
    g10 = graph_count(0.95,1.5,res)
        
    output = output + dp_url + "_flag_sep_esc_"                   #0                                                      
    output = output + user_name + "_flag_sep_esc_"                      #1
    # output = output + dp_url + "_flag_sep_esc_"   
    output = output + desc + "_flag_sep_esc_"     #2
    # output = output + dp_url + "_flag_sep_esc_"     #
    output = output + str(follwer_count) + "_flag_sep_esc_"          #3
    output = output + str(following_count) + "_flag_sep_esc_"          #4
    output = output + str(created_at) + "_flag_sep_esc_"      #5
    output = output + str(per_index) + "_flag_sep_esc_"  #6
    output = output + str(col_index) + "_flag_sep_esc_" #7
    output = output + str(retweet_perc) + "_flag_sep_esc_"   #8
    output = output + str(noTweet) + "_flag_sep_esc_"   #9
    # output = output + str(following_count) + "_flag_sep_esc_"   
    output = output + str(keyPTweet) + "_flag_sep_esc_"          #10
    output = output + str(clusters) + "_flag_sep_esc_"      #11
    output = output + str(TPcluster) + "_flag_sep_esc_"     #12
    output = output + str(noTweetPDay) + "_flag_sep_esc_"    #13
    # output = output + str(following_count) + "_flag_sep_esc_"      
    output = output + str(g0) + "##" + str(g1) + "##" + str(g2) + "##" + str(g3) + "##" + str(g4) + "##" + str(g5) + "##" + str(g6) + "##" + str(g7) + "##" + str(g8) + "##" + str(g9) + "##" + str(g10) + "_flag_sep_esc_"
    output = output + str(genu) + "##" + str(collusive) + "##" + str(unsure) + "_flag_sep_esc_"  #15
    output = output + str(week0) + "##" + str(week1) + "##" + str(week2) + "##" + str(week3) + "##" + str(week4) + "##" + str(week5) + "##" + str(week5) + "_flag_sep_esc_"
    output = output + listjoin(un) + "_flag_sep_esc_"  #17
    output = output + listjoin(res) + "_flag_sep_esc_"  #18
    output = output + listjoin(sn) + "_flag_sep_esc_" #19
    output = output + listjoin(dp) + "_flag_sep_esc_"  #20
    # print(output)
    # print(5)
        
    
    return output

# z = ann.predict(sc.transform([user])) print(z)
#1 genu
#0 fake

# Create your views here.
def index(request):
    # print(settings.BASE_DIR)
    return render(request,'index.html')

def dashboard(request):
    
    return render(request,'dashboard.html')

def neural(request):
    screen_name = request.POST.get("sn", "")
    response = complete_user_analysis(screen_name)
    return HttpResponse(response)