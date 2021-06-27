import Loader
import string
import Text_Preprocess
from shutil import copyfile
import os
import requests
from global_variable import dataset_dir, root_dir
from bleu_score import find_matched_image

# sentence = input('enter the sentence you would like to get images for !')

sentence = 'A blog (a truncation of "weblog")[1] is a discussion or informational website published on the World Wide Web consisting of discrete, often informal diary-style text entries (posts). Posts are typically displayed in reverse chronological order, so that the most recent post appears first, at the top of the web page. Until 2009, blogs were usually the work of a single individual,[citation needed] occasionally of a small group, and often covered a single subject or topic. In the 2010s, "multi-author blogs" (MABs) emerged, featuring the writing of multiple authors and sometimes professionally edited. MABs from newspapers, other media outlets, universities, think tanks, advocacy groups, and similar institutions account for an increasing quantity of blog traffic. The rise of Twitter and other "microblogging" systems helps integrate MABs and single-author blogs into the news media. Blog can also be used as a verb, meaning to maintain or add content to a blog.' \
           'The emergence and growth of blogs in the late 1990s coincided with the advent of web publishing tools that facilitated the posting of content by non-technical users who did not have much experience with HTML or computer programming. Previously, a knowledge of such technologies as HTML and File Transfer Protocol had been required to publish content on the Web, and early Web users therefore tended to be hackers and computer enthusiasts. In the 2010s, the majority are interactive Web 2.0 websites, allowing visitors to leave online comments, and it is this interactivity that distinguishes them from other static websites.[2] In that sense, blogging can be seen as a form of social networking service. Indeed, bloggers not only produce content to post on their blogs but also often build social relations with their readers and other bloggers.[3] However, there are high-readership blogs which do not allow comments.' \
           'Many blogs provide commentary on a particular subject or topic, ranging from philosophy, religion, and arts to science, politics, and sports. Others function as more personal online diaries or online brand advertising of a particular individual or company. A typical blog combines text, digital images, and links to other blogs, web pages, and other media related to its topic. The ability of readers to leave publicly viewable comments, and interact with other commenters, is an important contribution to the popularity of many blogs. However, blog owners or authors often moderate and filter online comments to remove hate speech or other offensive content. Most blogs are primarily textual, although some focus on art (art blogs), photographs (photoblogs), videos (video blogs or "vlogs"), music (MP3 blogs), and audio (podcasts). In education, blogs can be used as instructional resources; these are referred to as edublogs. Microblogging is another type of blogging, featuring very short posts.'

url = 'https://dev.api.videowiki.pt/api/sentence_detection/'
myobj = {'text': sentence, 'break_type': 'short'}

json_data = requests.post(url, data=myobj).json()

for index in json_data['sentences'].keys():
    print(index, json_data['sentences'][index])

idx = str(input('enter the index for which you like to get images!'))
predicted_description = json_data['sentences'][idx]
table = str.maketrans('', '', string.punctuation)

desc = predicted_description.split()
desc = [word.lower() for word in desc]
desc = [word.translate(table) for word in desc]
desc = [word for word in desc if len(word) > 1]
desc = [word for word in desc if word.isalpha()]
predicted_description = ' '.join(desc)
print(predicted_description)

database_image_list = dataset_dir + 'Flickr8k_text/Flickr_8k.trainImages.txt'
image_caption_list = Loader.load_set(database_image_list)

matchedimages_filelist = find_matched_image(
    image_caption_list,
    predicted_description
)

print(len(matchedimages_filelist))
flickr_dataset_dir = dataset_dir + 'Flickr8k_Dataset/Flicker8k_Dataset/'
matched_img_txt = open(root_dir + 'upload/matched_images.txt', "w")
matched_img_dir = root_dir + 'upload/matched-images'

for the_file in os.listdir(matched_img_dir):
    file_path = os.path.join(matched_img_dir, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

desc_text = Text_Preprocess.load_text(
    dataset_dir + 'Flickr8k_text/Flickr8k.token.txt'
)
descriptions = Text_Preprocess.load_description(
    desc_text
)

i = 0
for img in matchedimages_filelist:
    img_path = flickr_dataset_dir + img + '.jpg'
    i += 1
    matched_img_txt.write(descriptions[img][0] + '\n')
    copyfile(img_path, matched_img_dir + '/' + format(i, '03d') + '.jpg')

matched_img_txt.close()
