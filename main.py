import instaloader
from datetime import date
import getpass
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

'''PATH = "C:\Program Files (x86)\chromedriver.exe"
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

driver = webdriver.Chrome(PATH, service=Service(ChromeDriverManager().install()), options=chrome_options)
driver.get("https://instagram.com")

first_div = driver.find_element(By.ID, "react-root")
first_section = first_div.find_element(By.CLASS_NAME, '_9eogI')
first_main = first_section.find_element(By.CLASS_NAME, 'wG4fl')
print(first_main, "HERE")

driver.quit()'''

ig_loader = instaloader.Instaloader()
# Logging in to account
username = input("Enter username: ")
password = getpass.getpass("Enter password: ")
print("Logging in...")
ig_loader.login(username, password)
# Getting profile
print("Getting user profile...")
profile = instaloader.Profile.from_username(ig_loader.context, username)

# If a username from curr_list is not in compare_list, then append that username to the follower_changes_list
def getFollowerChanges(curr_list, compare_list):
    follower_changes_list = []
    for user in curr_list:
        if user not in compare_list:
            follower_changes_list.append(user.strip())
    return follower_changes_list

# Print usernames that have followed/unfollowed (action) user since the original date.
def displayData(follower_list, action, original_date):
    if len(follower_list) == 0:
        print("\nNo users have", action, "since", original_date)
    else:
        print("\nUsers that have", action, "since", original_date + ":", follower_list)
        print("Number of users that have", action + ":", len(follower_list))

# Fill FollowerFile.txt with the data from NewFollower.txt, update any dates as necessary.
def swapFileData():
    swap_data = input("Would you like to update your old follower count with the current follower data? (y/n): ")
    if swap_data == "y":
        with open("NewFollowerFile.txt", "r") as new_followers_file, open("FollowerFile.txt", "w+") as curr_follower_file:
            for line in new_followers_file:
                curr_follower_file.write(line)

# Function that outputs new followers/unfollowers from original date.
def followerData():
    # Getting list of user's followers
    with open("FollowerFile.txt", "a+") as follower_file, open("NewFollowerFile.txt", "w+") as new_follower_file:
        hasData = False
        follower_file.seek(0)
        data = follower_file.readlines()
        # If data's len is 0, then FollowerFile.txt has nothing in it, and it must be filled with current follower data.
        # Otherwise, the program will populate NewFollowerFile.txt with present follower data.
        if len(data) > 0:
            curr_file = new_follower_file
            curr_file.write("Followers list from " + str(date.today()) + "\n")
            hasData = True
        else:
            follower_file.write("Followers list from " + str(date.today()) + "\n")
            curr_file = follower_file
        print("Getting user's followers list...")
        for follower in profile.get_followers():
            curr_file.write(follower.username + "\n")
    # If FollowerFile.txt already has data in it, then find profiles that have followed/unfollowed user.
    if hasData:
        with open("FollowerFile.txt", "r") as follower_file, open("NewFollowerFile.txt", "r") as new_follower_file:
            # Comparing the usernames from both text files.
            original_date, curr_date = next(follower_file), next(new_follower_file)
            new_followers_list = new_follower_file.read().split("\n")
            old_followers_list = follower_file.read().split("\n")
            # Store new followers/unfollowers
            unfollow_list = getFollowerChanges(old_followers_list, new_followers_list)
            new_follower_list = getFollowerChanges(new_followers_list, old_followers_list)
            # Output new profiles that have followed/unfollowed user, and prompt if FollowerFile.txt should be updated.
            displayData(unfollow_list, "unfollowed", original_date[20: len(original_date) - 1])
            displayData(new_follower_list, "followed", original_date[20: len(original_date) - 1])
            swapFileData()

# Iterate through the following list of user, then search for user being present in the followee's following list.
def getUnloyalFollowing():
    unloyal_list = []
    print("Going through user's following list...")
    for followee in profile.get_followees():
        isFollowingBack = False
        curr_profile = instaloader.Profile.from_username(ig_loader.context, followee.username)
        print("Processing", followee.username)
        for f in curr_profile.get_followees():
            if f.username == username:
                isFollowingBack = True
                break
        if not isFollowingBack:
            unloyal_list.append(curr_profile.username)
    return unloyal_list

followerData()
# print(getUnloyalFollowing())
