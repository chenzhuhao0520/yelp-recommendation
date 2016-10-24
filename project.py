import json
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
filepath = 'review.json'#yelp_academic_dataset_
decoder = json.JSONDecoder()
jsonopen = open('review.json')
with open('review.json') as f:
    reviews = pd.DataFrame(json.loads(line) for line in f)


decoder = json.JSONDecoder()
jsonopen = open('yelp_academic_dataset_user.json')
with open('yelp_academic_dataset_user.json') as f:
    users = pd.DataFrame(json.loads(line) for line in f)

decoder = json.JSONDecoder()
jsonopen = open('yelp_academic_dataset_business.json')
with open('yelp_academic_dataset_business.json') as f:
    business = pd.DataFrame(json.loads(line) for line in f)


data = pd.merge(business, reviews, on = 'business_id', how = 'inner')

#data.head(2)

newdata = pd.merge(data, users, on = 'user_id', how = 'inner')


newdata['categories_str'] = newdata['categories'].map(str)
newdata = newdata.loc[newdata['categories_str'].str.contains('Restaurants',na=False), :]
#print newdata.head(10)

joinedFrames= newdata[ newdata.city == "Las Vegas"][['business_id', 'name_x', 'review_count_x', 'stars_x', 'user_id', 'name_y', 'review_count_y', 'stars_y', 'city', 'average_stars','review_count_y', 'review_id']]
new_columns = joinedFrames.columns.values
new_columns[1] = 'biz_name'
new_columns[2] = 'business_review_count'
new_columns[3] = 'business_average'
new_columns[5] = 'user_name'
new_columns[6] = 'user_review_count'
new_columns[7] = 'stars'
new_columns[9] = 'user_avg'
new_columns[10] ='del'
joinedFrames.columns = new_columns
joinedFrames = joinedFrames.drop('del', 1)

#print joinedFrames

def recompute_frame(ldf):
    """
    takes a dataframe ldf, makes a copy of it, and returns the copy
    with all averages and review counts recomputed
    this is used when a frame is subsetted.
    """
    ldfu=ldf.groupby('user_id')
    ldfb=ldf.groupby('business_id')
    user_avg=ldfu.stars.mean()
    user_review_count=ldfu.review_id.count()
    business_avg=ldfb.stars.mean()
    business_review_count=ldfb.review_id.count()
    nldf=ldf.copy()
    nldf.set_index(['business_id'], inplace=True)
    nldf['business_avg']=business_avg
    nldf['business_review_count']=business_review_count
    nldf.reset_index(inplace=True)
    nldf.set_index(['user_id'], inplace=True)
    nldf['user_avg']=user_avg
    nldf['user_review_count']=user_review_count
    nldf.reset_index(inplace=True)
    return nldf

smallidf = joinedFrames[(joinedFrames.user_review_count > 60) & (joinedFrames.business_review_count > 150)]
smalldf=recompute_frame(smallidf)
print("smalldf succeed\n")


def pearson_sim(rest1_reviews, rest2_reviews, n_common):
    """
    Given a subframe of restaurant 1 reviews and a subframe of restaurant 2 reviews,
    where the reviewers are those who have reviewed both restaurants, return 
    the pearson correlation coefficient between the user average subtracted ratings.
    The case for zero common reviewers is handled separately. Its
    ok to return a NaN if any of the individual variances are 0.
    """
    if n_common==0:
        rho=0.
    else:
        diff1=rest1_reviews['stars']-rest1_reviews['user_avg']
        diff2=rest2_reviews['stars']-rest2_reviews['user_avg']
        rho=pearsonr(diff1, diff2)[0]
    return rho

def get_restaurant_reviews(restaurant_id, df, set_of_users):
    """
    given a resturant id and a set of reviewers, return the sub-dataframe of their
    reviews.
    """
    mask = (df.user_id.isin(set_of_users)) & (df.business_id==restaurant_id)
    reviews = df[mask]
    reviews = reviews[reviews.user_id.duplicated()==False]
    return reviews 


def calculate_similarity(rest1, rest2, df, similarity_func):
    # find common reviewers
    rest1_reviewers = df[df.business_id==rest1].user_id.unique()
    rest2_reviewers = df[df.business_id==rest2].user_id.unique()
    common_reviewers = set(rest1_reviewers).intersection(rest2_reviewers)
    n_common=len(common_reviewers)
    #get reviews
    rest1_reviews = get_restaurant_reviews(rest1, df, common_reviewers)
    rest2_reviews = get_restaurant_reviews(rest2, df, common_reviewers)
    sim=similarity_func(rest1_reviews, rest2_reviews, n_common)
    if np.isnan(sim):
        return 0, n_common
    return sim, n_common



class Database:
    "A class representing a database of similaries and common supports"
    
    def __init__(self, df):
        "the constructor, takes a reviews dataframe like smalldf as its argument"
        database={}
        self.df=df
        self.uniquebizids={v:k for (k,v) in enumerate(df.business_id.unique())}
        keys=self.uniquebizids.keys()
        l_keys=len(keys)
        self.database_sim=np.zeros([l_keys,l_keys])
        self.database_sup=np.zeros([l_keys, l_keys], dtype=np.int)
        
    def populate_by_calculating(self, similarity_func):
        """
        a populator for every pair of businesses in df. takes similarity_func like
        pearson_sim as argument
        """
        items=self.uniquebizids.items()
        for b1, i1 in items:
            for b2, i2 in items:
                if i1 < i2:
                    sim, nsup=calculate_similarity(b1, b2, self.df, similarity_func)
                    self.database_sim[i1][i2]=sim
                    self.database_sim[i2][i1]=sim
                    self.database_sup[i1][i2]=nsup
                    self.database_sup[i2][i1]=nsup
                elif i1==i2:
                    nsup=self.df[self.df.business_id==b1].user_id.count()
                    self.database_sim[i1][i1]=1.
                    self.database_sup[i1][i1]=nsup
                    

    def get(self, b1, b2):
        
        sim=self.database_sim[self.uniquebizids[b1]][self.uniquebizids[b2]]
        nsup=self.database_sup[self.uniquebizids[b1]][self.uniquebizids[b2]]
        return (sim, nsup)


db=Database(smalldf)
db.populate_by_calculating(pearson_sim)

#print(smalldf.user_id)
print("step2 succeed\n")

def shrunk_sim(sim, n_common, reg=3.):
    "takes a similarity and shrinks it down by using the regularizer"
    ssim=(n_common*sim)/(n_common+reg)
    return ssim


from operator import itemgetter
def knearest(restaurant_id, set_of_restaurants, dbase, k=7, reg=3.):
    """
    Given a restaurant_id, dataframe, and database, get a sorted list of the
    k most similar restaurants from the entire database.
    """
    similars=[]
    for other_rest_id in set_of_restaurants:
        if other_rest_id!=restaurant_id:
            sim, nc=dbase.get(restaurant_id, other_rest_id)
            ssim=shrunk_sim(sim, nc, reg=reg)
            similars.append((other_rest_id, ssim, nc ))
    similars=sorted(similars, key=itemgetter(1), reverse=True)
    return similars[0:k]

"""
testbizid="lliksv-tglfUz1T3B3vgvA"
testbizid2="q_DrPmiLrHEpR_SQvQXELQ"
testbizid4="cMBZ46gNJPYY7zkxDb1piw"
testbizid5="T4AyBSffvi3prLrw0H2enA"
"""
def biznamefromid(df, theid):
    return df['biz_name'][df['business_id']==theid].values[0]
def usernamefromid(df, theid):
    return df['user_name'][df['user_id']==theid].values[0]
"""
print testbizid, biznamefromid(smalldf,testbizid)
print testbizid2, biznamefromid(smalldf, testbizid2)
print testbizid4, biznamefromid(smalldf, testbizid4)

tops=knearest(testbizid, smalldf.business_id.unique(), db, k=7, reg=3.)
print "For ", biznamefromid(smalldf, testbizid), ", top matches are:"
for i, (biz_id, sim, nc) in enumerate(tops):
    print i,biz_id, "| Sim", sim, "| Support",nc

print("business knearest succeed!\n")
"""
def get_user_top_choices(user_id, df, numchoices=5):
    "get the sorted top 5 restaurants for a user by the star rating the user gave them"
    udf=df[df.user_id==user_id][['business_id','stars']].sort(['stars'], ascending=False).head(numchoices)
    return udf
"""
testuserid="072tu2_MFWmw-jb1O6DQYg"
testuserid2="FLxgiNF9rsiXR-zbEcWBRQ"
testuserid3="IPYblqP7d2NE9pvHLViPOA"
testuserid4=smalldf.user_id[1000].encode('ascii','ignore')
testuserid5=smalldf.user_id[20].encode('ascii','ignore')
print "For user", usernamefromid(smalldf,testuserid), "top choices are:"
bizs=get_user_top_choices(testuserid, smalldf)['business_id'].values
[biznamefromid(smalldf, biz_id) for biz_id in bizs]
"""
print("next step: top recos for user:")

def get_top_recos_for_user(userid, df, dbase, n=5, k=7, reg=3.):
    bizs=get_user_top_choices(userid, df, numchoices=n)['business_id'].values
    rated_by_user=df[df.user_id==userid].business_id.values
    tops=[]
    for ele in bizs:
        t=knearest(ele, df.business_id.unique(), dbase, k=k, reg=reg)
        for e in t:
            if e[0] not in rated_by_user:
                tops.append(e)

    #there might be repeats. unique it
    ids=[e[0] for e in tops]
    uids={k:0 for k in list(set(ids))}

    topsu=[]
    for e in tops:
        if uids[e[0]] == 0:
            topsu.append(e)
            uids[e[0]] =1
    topsr=[]     
    for r, s,nc in topsu:
        avg_rate=df[df.business_id==r].stars.mean()
        topsr.append((r,avg_rate))
        
    topsr=sorted(topsr, key=itemgetter(1), reverse=True)

    if n < len(topsr):
        return topsr[0:n]
    else:
        return topsr



#Defining the predicted rating
def knearest_amongst_userrated(restaurant_id, user_id, df, dbase, k=7, reg=3.):
    dfuser=df[df.user_id==user_id]
    bizsuserhasrated=dfuser.business_id.unique()
    return knearest(restaurant_id, bizsuserhasrated, dbase, k=k, reg=reg)

def rating(df, dbase, restaurant_id, user_id, k=7, reg=3.):
    mu=df.stars.mean()
    users_reviews=df[df.user_id==user_id]
    nsum=0.
    scoresum=0.
    nears=knearest_amongst_userrated(restaurant_id, user_id, df, dbase, k=k, reg=reg)
    restaurant_mean=df[df.business_id==restaurant_id].business_avg.values[0]
    user_mean=users_reviews.user_avg.values[0]
    scores=[]
    for r,s,nc in nears:
        scoresum=scoresum+s
        scores.append(s)
        r_reviews_row=users_reviews[users_reviews['business_id']==r]
        r_stars=r_reviews_row.stars.values[0]
        r_avg=r_reviews_row.business_avg.values[0]
        rminusb=(r_stars - (r_avg + user_mean - mu))
        nsum=nsum+s*rminusb
    baseline=(user_mean +restaurant_mean - mu)
    #we might have nears, but there might be no commons, giving us a pearson of 0
    if scoresum > 0.:
        val =  nsum/scoresum + baseline
    else:
        val=baseline
    return val
#Testing the ratings
def get_other_ratings(restaurant_id, user_id, df):
    "get a user's rating for a restaurant and the restaurant's average rating"
    choice=df[(df.business_id==restaurant_id) & (df.user_id==user_id)]
    users_score=choice.stars.values[0]
    average_score=choice.business_avg.values[0]
    return users_score, average_score

"""
testuserid="072tu2_MFWmw-jb1O6DQYg"
testuserid2="FLxgiNF9rsiXR-zbEcWBRQ"
testuserid3="IPYblqP7d2NE9pvHLViPOA"
testuserid4=smalldf.user_id[1000].encode('ascii','ignore')
testuserid5=smalldf.user_id[20].encode('ascii','ignore')
print "For user", usernamefromid(smalldf,testuserid), "top choices are:" 

bizs=get_user_top_choices(testuserid, smalldf)['business_id'].values
[biznamefromid(smalldf, biz_id) for biz_id in bizs]


print "For user", usernamefromid(smalldf,testuserid), "the top recommendations are:"
toprecos=get_top_recos_for_user(testuserid, smalldf, db, n=5, k=7, reg=3.)
for biz_id, biz_avg in toprecos:
    print biznamefromid(smalldf,biz_id), "| Average Rating |", biz_avg

print "For user", usernamefromid(smalldf,testuserid2), "the top recommendations are:"
toprecos=get_top_recos_for_user(testuserid2, smalldf, db, n=5, k=7, reg=3.)
for biz_id, biz_avg in toprecos:
    print biznamefromid(smalldf,biz_id), "| Average Rating |", biz_avg


print "User Average", smalldf[smalldf.user_id==testuserid].stars.mean(),"for",usernamefromid(smalldf,testuserid)
print "Predicted ratings for top choices calculated earlier:"
for biz_id,biz_avg in toprecos:
    print biznamefromid(smalldf, biz_id),"|",rating(smalldf, db, biz_id, testuserid, k=7, reg=3.),"|","Average",biz_avg 

print "User Average", smalldf[smalldf.user_id==testuserid2].stars.mean(),"for",usernamefromid(smalldf,testuserid2)
print "Predicted ratings for top choices calculated earlier:"
for biz_id,biz_avg in toprecos:
    print biznamefromid(smalldf, biz_id),"|",rating(smalldf, db, biz_id, testuserid2, k=7, reg=3.),"|","Average",biz_avg 

print "for user",usernamefromid(smalldf,testuserid), 'avg', smalldf[smalldf.user_id==testuserid].stars.mean() 
for biz_id in bizs:
    print "----------------------------------"
    print biznamefromid(smalldf, biz_id)
    print "Predicted Rating:",rating(smalldf, db, biz_id, testuserid, k=7, reg=3.) 
    u,a=get_other_ratings(biz_id, testuserid, smalldf)
    print "Actual User Rating:",u,"Avg Rating",a
"""
"""
write_file = open("recommend_for_user.json", "a")
for one_user_id in smalldf.user_id.unique():
    bizs=get_user_top_choices(one_user_id, smalldf)['business_id'].values
    #[biznamefromid(smalldf, biz_id) for biz_id in bizs]
    recommendations = ''
    i = 0
    for biz_id in bizs:
        biz_name = biznamefromid(smalldf, biz_id)
        u,a=get_other_ratings(biz_id, one_user_id, smalldf)
        predicted_rate = rating(smalldf, db, biz_id, one_user_id, k=7, reg=3.) 
        if i < len(bizs)-1:
            recommendations = recommendations + '{' + '"biz_id": "' + biz_id + '", "biz_name": "' + biz_name + '", "avg_rate": ' + str(a) + ', "actual_rate": ' + str(u) + ', "predicted_rate": ' + str(predicted_rate) + '}, '
        else:
            recommendations = recommendations + '{' + '"biz_id": "' + biz_id + '", "biz_name": "' + biz_name + '", "avg_rate": ' + str(a) + ', "actual_rate": ' + str(u) + ', "predicted_rate": ' + str(predicted_rate) + '}'
        i +=1
	#line = {user_id: id, user_name: name, recommendations: { {b_id: , b_name: , avg_rate: int, predicted_rate: double, actual_rate: int}, {}  }}
    line = '{'+'"user_id": "'+one_user_id + '", "user_name": "'+usernamefromid(smalldf, one_user_id) + '", "recomend": ['+ recommendations + ']}, '
    write_file.write(line.encode('utf-8'))






"""

j = 0
biz_file = open("similar_biz.json", "a")
for business_id in smalldf.business_id.unique():
    tops=knearest(business_id, smalldf.business_id.unique(), db, k=7, reg=3.)
    similar_business = ''
    for i, (biz_id, sim, nc) in enumerate(tops):
        #print biznamefromid(smalldf,biz_id), "| Sim", sim, "| Support",nc
        if i < 6:
            similar_business += '{' + '"business_id": "' + biz_id + '", "business_name": "' + biznamefromid(smalldf,biz_id) + '", "sim": ' + str(sim) + ', "support": ' + str(nc) + '}, '
        else:
            similar_business += '{' + '"business_id": "' + biz_id + '", "business_name": "' + biznamefromid(smalldf,biz_id) + '", "sim": ' + str(sim) + ', "support": ' + str(nc) + '}'
    line = '{' + 'biz_id: ' +  business_id + ', biz_name: ' + biznamefromid(smalldf, business_id) + ', similar_biz: ' + '{' + similar_business + '}' + '}'
    biz_file.write(line.encode('utf-8'))
    j += 1
    if j > 5000:
        break;

"""

