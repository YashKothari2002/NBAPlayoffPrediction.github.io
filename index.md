<html>
<head>
<title>LaTeX4Web 1.4 OUTPUT</title>
<style type="text/css">
<!--
 body {color: black;  background-color:#FFCC99;  }
 div.p { margin-top: 7pt;}
 td div.comp { margin-top: -0.6ex; margin-bottom: -1ex;}
 td div.comb { margin-top: -0.6ex; margin-bottom: -.6ex;}
 td div.norm {line-height:normal;}
 td div.hrcomp { line-height: 0.9; margin-top: -0.8ex; margin-bottom: -1ex;}
 td.sqrt {border-top:2 solid black;
          border-left:2 solid black;
          border-bottom:none;
          border-right:none;}
 table.sqrt {border-top:2 solid black;
             border-left:2 solid black;
             border-bottom:none;
             border-right:none;}
-->
</style>
<meta http-equiv="Content-Type" content="text/html; charset=iso-latin-1"/>
</head>
<body>

\documentclassarticle

\usepackagemicrotype
\usepackagegraphicx
\usepackagesubfigure
\usepackagebooktabs 
\usepackagehyperref

\newcommand\theHalgorithm\arabicalgorithm

\usepackage[accepted]styles/icml2020


<font face=symbol>i</font>cmltitlerunningPredicting the ranked NBA playoff bracket
\begindocument

\twocolumn[
<font face=symbol>i</font>cmltitlePredicting NBA Playoffs Ranked Bracket 




\beginicmlauthorlist
<font face=symbol>i</font>cmlauthorEkisha Basu
<font face=symbol>i</font>cmlauthorConnor Caldwell
<font face=symbol>i</font>cmlauthorYash Kothari
<font face=symbol>i</font>cmlauthorPranav Putta
<font face=symbol>i</font>cmlauthorLakshman Ravoori
\endicmlauthorlist

<font face=symbol>i</font>cmlcorrespondingauthorPranav Puttapputta7@gatech.edu

<font face=symbol>i</font>cmlkeywordsMachine Learning, ICML

\vskip 0.3in
]



Phys.Rev. intAffiliationsAndNotice<font face=symbol>i</font>cmlEqualContribution 
\beginabstract
This project aims to use a deep neural network to predict the NBA playoffs<font face=symbol>¢</font> ranked bracket. The model will be tested against 2 baselines (gradient boosting and a randomized algorithm). The majority of previous approaches to NBA predictions have been focused on generating the playoff spread. This project differentiates itself by focusing on the ranked bracket for the playoffs. At the end of this project, we aim to be able to predict the ranked bracket for the 2021 playoffs. \endabstract

<p><a name="toc.1"><h1>1&nbsp;Introduction</h1>
<a name="refsubmission">


The NBA is one of the top grossing sports in the United States, bringing in over \8 billion in revenue per year. As such, sports betting is a prominent aspect of American culture that engages millions of people nationwide.
Engineering a prediction algorithm that can reliably produce insights about the outcomes of games would be of significance to consumers, the NBA franchise, opposition team analysis, and the gambling industry.

As is common with any sport, the NBA has a dense collection of data and statistics ranging from points scored all the way to the number of blocks, assists, and steals made by every player. The unpredictability of the NBA is very likely due to our inability to accurately quantify players and performance metrics on the field. Through this project, we hope to understand which performance metrics and statistics are the major contributors to the overall game outcome.
<p><a name="toc.2"><h1>2&nbsp;Problem Definition</h1>
The overarching problem can be defined as a decision tree and ranking problem.
<p><a name="toc.2.1"><h2>2.1&nbsp;Application of Decision Tree and Ranking</h2>
Each individual basketball game over the course of an NBA season can be represented as a decision tree. There are <font face=symbol>¢</font>N<font face=symbol>¢</font> number of input features <font face=symbol>-</font> counting statistics <font face=symbol>-</font> that can be traversed over to eventually reach a binary outcome <font face=symbol>-</font> Win for Team A or Win for Team B. Taking the sum over all results over a season for an arbitrary team <font face=symbol>¢</font>A<font face=symbol>¢</font>, we can then "rank" teams in order of the number of total wins they accumulate. 
<p><a name="toc.2.2"><h2>2.2&nbsp;Input Space</h2>
Basketball statistics data will be pulled from basketball<font face=symbol>-</font>reference.com and espn.com/basketball/stats. Unique dataset will be created after performing statistical analysis of correlation between various stats and the impact on team win. Final dataset will be pre<font face=symbol>-</font>processed using principal component analysis (PCA) and mean<font face=symbol>-</font>centering to reduce dimensionality. 
<p><a name="toc.2.3"><h2>2.3&nbsp;Output Space</h2>
(x1, x2, x3, x4, x5, x6, x7, x8)
(x1_A,x2_A,x3_A,x4_A,x5_A,x6_A,x7_a,x8_A)  \newline
2x8 Dimensional Matrix <font face=symbol>-</font> ranked from x1<font face=symbol>-</font>x8 for each conference (east and west)
<p><a name="toc.2.4"><h2>2.4&nbsp;Assumptions</h2>
a. Team win percentage is based solely on individual performances and quantifiable statistics \newline
b. Injuries to players unaccounted for (pre<font face=symbol>-</font>season predictions)

<p><a name="toc.3"><h1>3&nbsp;Data</h1>
<p><a name="toc.3.1"><h2>3.1&nbsp;Collection</h2>
To acquire the data, we decided to use data from ESPN and Basketball Reference. ESPN is a popular cable sports channel and Basketball Reference is one of the most popular databases for NBA history. As both of these sources are known to be reputable, we felt quite comfortable using them. Using Beautiful Soup python library we decided to webscrape NBA Regular Season Team Statistics and the NBA Playoffs Team Stats from the past 19 seasons (2001<font face=symbol>-</font>2002 season to the 2019<font face=symbol>-</font>2020 season). Novelty lies in combining the most impactful statistics on game results from multiple sources.

The data provided team statistics such as 3<font face=symbol>-</font>point percentage, free<font face=symbol>-</font>through percentage, points, assists, steals, etc for each season.

<p><a name="toc.3.2"><h2>3.2&nbsp;ANOVA (Analysis of Variance) Coefficient Analysis</h2>
The following table provides a small sample of the statistical analysis done on the input space to narrow down the set of all basketball statistics into an input space of meaningful (statistically significant) data that can be used to train a model for win prediction.

\begincenter
 \begintabular||c c c|| 
 \hline
 Feature Name & F<font face=symbol>-</font>test & Feature Selected<br>
 [0.5ex] 
 \hline\hline
 3pt \% & 13.47 & YES <br>
 
 \hline
 EFG \% & 18.36 & YES<br>

 \hline
 PTS & 12.20 & YES<br>

 \hline
 FT \% & 12.33 & YES<br>

 \hline
 FGM & 2.17 & NO<br>

 \hline
 MP & 0.05 & NO<br>

 \hline
 TO & 10.13<sup>*</sup> & YES<br>

 \hline
\endtabular
\endcenter
We see that there is a F<font face=symbol>-</font>test metric as well as a YES or NO binary classification for whether a feature was selected for our training (input) space or not. There is a way to threshold this F<font face=symbol>-</font>metric to either accept (take) or reject (not take) the null hypothesis from the ANOVA test. 

Rejecting the null hypothesis indicates that variance exists between the two groups which, for this paper, are the feature (group 1) and the result of the game (group 2). A value of the F<font face=symbol>-</font>test above a given threshold means that we would reject the null hypothesis for that particular feature and hence make it a part of the data<font face=symbol>-</font>set that we use to train the model. This threshold is taken from the F<font face=symbol>-</font>table, which is based on the feature space, degrees of freedom, etc. We have taken our threshold as 4.23 (derived from the F<font face=symbol>-</font>table directly). 

<p><a name="toc.3.3"><h2>3.3&nbsp;Feature Importance from Gradient Boosting Model</h2>
The first trained version of our gradient boosting algorithm was trained on the entire dataset that we scraped. The reason behind this was to allow us to utilize \textittensorflow<font face=symbol>¢</font>s provided feature importance ranking from the gradient boosting model itself. We felt that taking this into consideration along with our own independent statistical analysis might be a good way of double<font face=symbol>-</font>verifying the importance of any given feature \textitx.
\beginfigure[htp]
    \centering
     <font face=symbol>Î</font> cludegraphics[width=8cm]Screen Shot 2021<font face=symbol>-</font>04<font face=symbol>-</font>06 at 10.36.54 PM.jpeg
    <font face=symbol>Ç</font>tionSample Feature Importance Chart from our Gradient Boosting Model
    <a name="reffig:meminmem">

\endfigure

From the above, we see the contribution of different features to the predicted probability (of a win) for a given team \textitA. This allows us to rank features by their importance in terms of the model<font face=symbol>¢</font>s predictions. One important note to be taken with this is that the model has not seen a smaller subset of the data, therefore, it has not had the chance to learn on smaller subsets at this point. This means that it cannot determine whether when given different data, its predictions might be better. Hence, these rankings are weighted less than the ANOVA analysis that has been done independently by our research group. 
<p><a name="toc.4"><h1>4&nbsp;Methods</h1>
We plan to use two baseline models to compare against our own model.

<p><a name="toc.4.1"><h2>4.1&nbsp;Baseline 1: Randomized Algorithm</h2>
Since rank prediction is quite a difficult and different task than simply predicting the outcomes of games, we wanted to use a completely randomized algorithm to demonstrate the significance of our results through our deep learning model.

This randomized algorithm simulated each game during a season and randomly selected the outcome from a uniform probability distribution. After creating a prediction for the outcome of each game, a ranked matrix was constructed, ordered by each team<font face=symbol>¢</font>s percentage of wins for the season.


<p><a name="toc.4.2"><h2>4.2&nbsp;Baseline 2: Gradient Boosting</h2>
For our second baseline, we chose to use gradient boosting, which is an ensemble tree learning method. Through this method, we combine a series of weak tree prediction models to create a strong learning model. This type of model helps mitigate the problem of overfitting data which can be particularly troublesome with historical training data sets which may have large variances.

As opposed to the random forest model, boosting attacks the bias<font face=symbol>-</font>variance tradeoff by starting with weak models and sequentially boosts its performance by continuing to build new trees, where each new tree in the sequence tries to fix up where the previous one made the biggest mistakes. 

<p><a name="toc.4.3"><h2>4.3&nbsp;Our Model: Feedforward Deep Neural Network</h2>
The team<font face=symbol>¢</font>s rank will be classified via a feedforward deep neural network (DNN). The model will have 6 to 8 layers and consist of various layer types. Memory is important, as we will need to remember each individual game prediction and use it to base future predictions.

<p><a name="toc.4.3.1"><h3>4.3.1&nbsp;Architecture of Memory</h3>
\beginfigure[htp]
    \centering
     <font face=symbol>Î</font> cludegraphics[width=4cm]Memory Architecture.png
    <font face=symbol>Ç</font>tionMemory in Memory Block
    <a name="reffig:meminmem">

\endfigure
Instead of using a standard ST/LSTM unit, we will use a memory in memory unit, better for spatiotemporal predictions (our individual game predictions will need to be recorded based on both prediction space as well as time of prediction)


<p><a name="toc.4.4"><h2>4.4&nbsp;Performance Metrics/Loss Functions</h2>
Recurrent unit is required within the deep neural network architecture for feedback from memory. Loss functions used will be two<font face=symbol>-</font>fold: 
<p><a name="toc.4.4.1"><h3>4.4.1&nbsp;Mean<font face=symbol>-</font>Squared Error (MSE)</h3>
Distance between our predicted outcome of a game and the ground truth of the actual result.
<p><a name="toc.4.4.2"><h3>4.4.2&nbsp;Mean Average Error (MAE)</h3>
Generally used as a variant of an error function for classification/regression problems. This has been used by our baseline models in the past and hence makes it easy for a comparative basis.
<p><a name="toc.4.4.3"><h3>4.4.3&nbsp;Memory Feedback</h3>
The memory of predictions for results of past games and the error in those predictions will be recurrently fed into our deep network as double<font face=symbol>-</font>verification (since our network performs two functions simultaneously).
<p><a name="toc.4.4.4"><h3>4.4.4&nbsp;Rank Performance</h3>
The final output of our prediction algorithm produces a ranked matrix ordered by the percentage of wins each team had during a particular season. In order to compare predicted ranked brackets to the true rank, we use the KendallTau statistic which outputs the association between two ranked lists. We can use this metric to evaluate the final result and give it a score.
<p><a name="toc.4.4.5"><h3>4.4.5&nbsp;Past Performance</h3>
Currently, we have a ceiling in terms of accuracy of NBA game prediction at 74\% which is referenced later in this paper. (done by a third party group and not our research group personally). The goal of this project is to see how utilizing unique datasets along with recurrent memory units with a deep neural network architecture can impact the performance of a model on NBA game predictions.

<p><a name="toc.5"><h1>5&nbsp;Results and Conclusion</h1>
<p><a name="toc.5.1"><h2>5.1&nbsp;Potential Results</h2>
A bracket of playoff teams (generated in a visual form) would be an optimal representation of our results. This has been done in the past by various models and seems like a good baseline for our result representation. Our goal is to be able to use regular season win sums to predict the progression of the bracket through the NBA playoffs (predict the overall winner). A table/ranking column vector will also be a result we would like to display. 
\beginfigure[htp]
    \centering
     <font face=symbol>Î</font> cludegraphics[width=6cm]nba<font face=symbol>-</font>playoff<font face=symbol>-</font>bracket<font face=symbol>-</font>2019<font face=symbol>-</font>finals<font face=symbol>-</font>ftrjpg<sub>g</sub>smlx0m6tqtn123yk70q2oab4.jpg
    <font face=symbol>Ç</font>tionPotential Visual Representation
    <a name="reffig:bracket">

\endfigure

<p><a name="toc.5.2"><h2>5.2&nbsp;Baseline 1 Results: Randomized Algorithm</h2>
As is expected, we reported that during each season, the outcomes of games were classified correctly roughly 50\% of the time each season.

We then took the simulated game data and ranked them according to the percentage of games each team won during the season. We then ran the Kendall Tau ranked order correlation statistic to determine the similarity between the ranked predictions and true rank from each season.

The Kendall Tau statistic produces a result between [<font face=symbol>-</font>1, 1]. The closer to 1, the stronger the association between the two rankings, and the closer to <font face=symbol>-</font>1 is a negative association. We want our Kendall Tau statistic to be as close to 1 as possible. A reverse order is the worst possible arrangement, resulting in a score of <font face=symbol>-</font>1. The p<font face=symbol>-</font>value from Kendall Tau is a two sided p<font face=symbol>-</font>value where the hypothesis test is for a null hypothesis where there is no association between the ranked lists. We divide the p<font face=symbol>-</font>value by 2 because we<font face=symbol>¢</font>re only concerned with positive association.

\begincenter
 \begintabular||c c c|| 
 \hline
 Season & Ranked Correlation & p<font face=symbol>-</font>value <br>
 [0.5ex] 
 \hline\hline
 2018 & <font face=symbol>-</font>0.00229 & 0.5 <br>
 
 \hline
 2019 & <font face=symbol>-</font>0.135 & 0.151 <br>

 \hline
 2020 & <font face=symbol>-</font>0.0344 & 0.402<br>

 \hline
\endtabular
\endcenter

<p><a name="toc.5.2.1"><h3>5.2.1&nbsp;2018 Predicted Ranked Matrix</h3>
 <font face=symbol>Î</font> cludegraphics[scale=0.15]images/2018<sub>p</sub>redicted<sub>r</sub>anked<font face=symbol>-</font>random.png
<p><a name="toc.5.2.2"><h3>5.2.2&nbsp;2018 True Ranked Matrix</h3>
 <font face=symbol>Î</font> cludegraphics[scale=0.15]images/2018<sub>t</sub>rue<sub>r</sub>anked<font face=symbol>-</font>random.png

From these scores, we can see that the random algorithm was not very successful in creating any type of association between the predicted ranked list and the true ranked list. Since each correlation score is very close to zero and the p<font face=symbol>-</font>values are very high (0.5 is the max since we only consider the positive association), we can conclude that the randomized algorithm was not effective in creating correct predictions.


<p><a name="toc.5.3"><h2>5.3&nbsp;Baseline 2 Results: Gradient Boosted Trees</h2>
After cleaning the data, we trained the gradient boosting model with the following parameters using the BoostedTreesClassifier taken from the TensorFlow library.

\begincenter
 \begintabular||c c|| 
 \hline
 Parameter & Value <br>
 [0.5ex] 
 \hline\hline
 n trees & 50 <br>
 
 \hline
 max depth & 3<br>

 \hline
 n batches per layer & 1 <br>

 \hline
\endtabular
\endcenter

This model was trained by taking data from previous seasons<font face=symbol>¢</font> overall team statistics and using this data to inform the next season<font face=symbol>¢</font>s predictions. Each game is simulated again, with the BoostedTreesClassifier evaluating the prediction for each game. Finally, each team is ranked based on its percentage of wins over the course of the season.

The ranked matrix correlation scores are shown below, again calculated using the Kendall Tau algorithm.

\begincenter
 \begintabular||c c c|| 
 \hline
 Season & Ranked Correlation & p<font face=symbol>-</font>value <br>
 [0.5ex] 
 \hline\hline
 2018 & 0.172 & 0.099 <br>
 
 \hline
 2019 & 0.0666 & 0.31 <br>

 \hline
 2020 & 0.0437 & 0.325 <br>

 \hline
\endtabular
\endcenter

From the results for the gradient boosting algorithm, we can see a clear improvement over the randomized algorithm. While 2019 and 2020 don<font face=symbol>¢</font>t show as much of a positive association, we can see 2018 has a much higher correlation than seen with the random algorithm. Overall, we see that the gradient boosting algorithm was more successful in generating predictions.

By analyzing the ranked matricies, we see there<font face=symbol>¢</font>s strong clustering, with the true winners of the season generally at the top of the bracket while the losers are at the bottom. While the ordering is not perfect, the clustering shows that there was a lot of association in the correlations.

<p><a name="toc.5.3.1"><h3>5.3.1&nbsp;2018 Predicted Ranked Matrix</h3>
 <font face=symbol>Î</font> cludegraphics[scale=0.15]images/2018<sub>p</sub>redicted<sub>r</sub>anked.png
<p><a name="toc.5.3.2"><h3>5.3.2&nbsp;2018 True Ranked Matrix</h3>
 <font face=symbol>Î</font> cludegraphics[scale=0.15]images/2018<sub>t</sub>rue<sub>r</sub>anked.png
<p><a name="toc.5.3.3"><h3>5.3.3&nbsp;Performance Metrics for Model</h3>
\begincenter
 \begintabular||c c|| 
 \hline
 Metric & Value <br>
 [0.5ex] 
 \hline\hline
 accuracy & 0.6538394, <br>
 
 \hline
 accuracy<font face=symbol>-</font>baseline & 0.5826744 <br>

 \hline
 auc & 0.68861175 <br>

 \hline
 auc<font face=symbol>-</font>precision<font face=symbol>-</font>recall & 0.74310637<br>
 
 \hline
 mean<font face=symbol>-</font>squared<font face=symbol>-</font>loss & 0.62664056 <br>

 \hline
 loss & 0.62664056 <br>

 \hline
 precision & 0.6760858 <br>

 \hline
\endtabular
\endcenter

Our accuracy and precision values are decently high for a task that is so random in its outcome, simply because of the relative equality of basketball teams. The overall highest ever accuracy for an NBA game prediction task is 74\% which was achieved by a Stanford paper. Our goal in this project is to get as close to, if not surpass, the accuracy of that paper while minimizing the loss of our predictions. The usage of recurrent memory in our predictions is hypothesized to have a positive impact which will be demonstrated in the final paper. 
<p><a name="toc.5.4"><h2>5.4&nbsp;Conclusion</h2>
The project<font face=symbol>¢</font>s success will definitely be measured based on how well our model can learn to predict results. However, the overarching goal of this project is to be able to come up with a more comprehensive result of statistical features that impact the outcome of a game. The usage of data as a basis for information is prevalent in sports. If we are able to identify a dataset of basketball counting statistics that can be said to span the most important features impacting winning, this would be a huge success. 

So far, we<font face=symbol>¢</font>ve learned how to analyze a feature space and determine which features to utilize and have explored using a number of models to use for our baseline metric, such as the random forest and gradient boosting. Our first attempt with random forest showed poor performance results and through research we decided to use gradient boosting as a replacement.

Continuing on with the semester, our goal is to utilize our analysis of the feature space and our baseline results to develop our deep learning model and memory unit that can outperform the baseline results.

<p><a name="toc.6"><h1>6&nbsp;References</h1>
Cheng, Ge, et al. “Predicting the Outcome of NBA Playoffs Based on the Maximum Entropy Principle.” Entropy, vol. 18, no. 12, 2016, p. 450., doi:10.3390/e18120450.

Kohli, Ikjyot Singh. “Finding Common Characteristics Among NBA Playoff Teams: A Machine Learning Approach.” SSRN Electronic Journal, 2016, doi:10.2139/ssrn.2764396.

Hu, Feifang, and James V. Zidek. “Forecasting NBA Basketball Playoff Outcomes Using the Weighted Likelihood.” Institute of Mathematical Statistics Lecture Notes <font face=symbol>-</font> Monograph Series A Festschrift for Herman Rubin, 2004, pp. 385–395., doi:10.1214/lnms/1196285406.

reHOOPerate. “NBA Neural Networks: Quantifying Playoff Performance by Number of Games Won and Seeding.” Medium, Re<font face=symbol>-</font>HOOP*PER<font face=symbol>-</font>Rate, 21 Aug. 2020, medium.com/re<font face=symbol>-</font>hoop<font face=symbol>-</font>per<font face=symbol>-</font>rate/nba<font face=symbol>-</font>neural<font face=symbol>-</font>networks<font face=symbol>-</font>quantifying<font face=symbol>-</font>playoff<font face=symbol>-</font>performance<font face=symbol>-</font>by<font face=symbol>-</font>number<font face=symbol>-</font>of<font face=symbol>-</font>games<font face=symbol>-</font>won<font face=symbol>-</font>and<font face=symbol>-</font>seeding<font face=symbol>-</font>9b8e9d2f6a55. 
\enddocument


<hr>
<p><h1>Table Of Contents</h1>
<p><a href="#toc.1"><h1>1&nbsp;Introduction</h1></a>
<p><a href="#toc.2"><h1>2&nbsp;Problem Definition</h1></a>
<p><a href="#toc.2.1"><h2>2.1&nbsp;Application of Decision Tree and Ranking</h2></a>
<p><a href="#toc.2.2"><h2>2.2&nbsp;Input Space</h2></a>
<p><a href="#toc.2.3"><h2>2.3&nbsp;Output Space</h2></a>
<p><a href="#toc.2.4"><h2>2.4&nbsp;Assumptions</h2></a>
<p><a href="#toc.3"><h1>3&nbsp;Data</h1></a>
<p><a href="#toc.3.1"><h2>3.1&nbsp;Collection</h2></a>
<p><a href="#toc.3.2"><h2>3.2&nbsp;ANOVA (Analysis of Variance) Coefficient Analysis</h2></a>
<p><a href="#toc.3.3"><h2>3.3&nbsp;Feature Importance from Gradient Boosting Model</h2></a>
<p><a href="#toc.4"><h1>4&nbsp;Methods</h1></a>
<p><a href="#toc.4.1"><h2>4.1&nbsp;Baseline 1: Randomized Algorithm</h2></a>
<p><a href="#toc.4.2"><h2>4.2&nbsp;Baseline 2: Gradient Boosting</h2></a>
<p><a href="#toc.4.3"><h2>4.3&nbsp;Our Model: Feedforward Deep Neural Network</h2></a>
<p><a href="#toc.4.3.1"><h3>4.3.1&nbsp;Architecture of Memory</h3></a>
<p><a href="#toc.4.4"><h2>4.4&nbsp;Performance Metrics/Loss Functions</h2></a>
<p><a href="#toc.4.4.1"><h3>4.4.1&nbsp;Mean<font face=symbol>-</font>Squared Error (MSE)</h3></a>
<p><a href="#toc.4.4.2"><h3>4.4.2&nbsp;Mean Average Error (MAE)</h3></a>
<p><a href="#toc.4.4.3"><h3>4.4.3&nbsp;Memory Feedback</h3></a>
<p><a href="#toc.4.4.4"><h3>4.4.4&nbsp;Rank Performance</h3></a>
<p><a href="#toc.4.4.5"><h3>4.4.5&nbsp;Past Performance</h3></a>
<p><a href="#toc.5"><h1>5&nbsp;Results and Conclusion</h1></a>
<p><a href="#toc.5.1"><h2>5.1&nbsp;Potential Results</h2></a>
<p><a href="#toc.5.2"><h2>5.2&nbsp;Baseline 1 Results: Randomized Algorithm</h2></a>
<p><a href="#toc.5.2.1"><h3>5.2.1&nbsp;2018 Predicted Ranked Matrix</h3></a>
<p><a href="#toc.5.2.2"><h3>5.2.2&nbsp;2018 True Ranked Matrix</h3></a>
<p><a href="#toc.5.3"><h2>5.3&nbsp;Baseline 2 Results: Gradient Boosted Trees</h2></a>
<p><a href="#toc.5.3.1"><h3>5.3.1&nbsp;2018 Predicted Ranked Matrix</h3></a>
<p><a href="#toc.5.3.2"><h3>5.3.2&nbsp;2018 True Ranked Matrix</h3></a>
<p><a href="#toc.5.3.3"><h3>5.3.3&nbsp;Performance Metrics for Model</h3></a>
<p><a href="#toc.5.4"><h2>5.4&nbsp;Conclusion</h2></a>
<p><a href="#toc.6"><h1>6&nbsp;References</h1></a>
</body>
</html>
