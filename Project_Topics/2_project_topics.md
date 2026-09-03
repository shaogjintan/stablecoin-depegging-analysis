# DSE4101 Capstone Project in Data Science and Economics I

College of Humanities and Sciences
National University of Singapore


Semester 1, AY 2026/2027

# Domains and Topics **1 Digital Currencies**

## **1.1 Volatility Forecasting**


Volatility serves as the backbone of finance, acting both as a critical signal for investors
and as a fundamental input for various financial models. Its importance lies in its ability
to capture uncertainty—the core characteristic of financial systems. With the increasing
integration of global financial markets, prolonged periods of uncertainty have become a
defining feature, further elevating the role of volatility. As a proxy for risk, volatility is
indispensable in key financial fields such as asset pricing and risk management.


The estimation and forecasting of volatility have been at the heart of financial research
for decades. Traditional approaches, such as ARCH- and GARCH-type models, laid the
groundwork for this field. However, these models face challenges, including their inability to fully address issues like volatility clustering and information asymmetry. Recent
developments in financial markets, combined with advancements in machine learning,
have spurred a renewed interest in more sophisticated approaches to volatility prediction.
This evolution is particularly relevant in the digital currency domain, where the volatility
of cryptocurrencies is markedly higher and presents unique challenges and opportunities
for investors and analysts.


To dive into volatility forecasting, you can start from the following papers that survey
traditional and cutting-edge methodologies. These papers introduce key models and
their applications to cryptocurrency markets. From statistical techniques like GARCH to
advanced machine learning approaches such as support vector regression, these studies
highlight the evolution and effectiveness of forecasting tools in this domain


[• Christensen, K., Siggaard, M., & Veliyev, B. (2023). Machine learning approach to](https://doi.org/10.1093/jjfinec/nbac020)
[volatility forecasting.](https://doi.org/10.1093/jjfinec/nbac020) _Journal of Financial Econometrics_, 21(5), 1680-1727.


[• Dudek, Grzegorz, Piotr Fiszeder, PawełKobus, and Witold Orzeszko (2024). Fore-](https://doi.org/10.1016/j.asoc.2023.111132)
[casting cryptocurrencies volatility using statistical and machine learning methods:](https://doi.org/10.1016/j.asoc.2023.111132)
[A comparative study.](https://doi.org/10.1016/j.asoc.2023.111132) _Applied Soft Computing_ 151: 111132.
Data: `https://repod` _._ `icm` _._ `edu` _._ `pl/dataset` _._ `[xhtml?persistentId=doi:10](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/YVS7HE)` _._ `18150/YVS7HE`


1


[• Mashrur, A., Luo, W., Zaidi, N. A., & Robles-Kelly, A. (2020). Machine learning for](https://doi.org/10.1109/ACCESS.2020.3036322)
[financial risk management: A survey. Ieee Access, 8, 203203-203223.](https://doi.org/10.1109/ACCESS.2020.3036322)
Read: Section IV.A.1 Volatility Measurement and Forecasting.


Christensen et al. examine the predictive power of machine learning methods, such
as regularization and regression trees, compared to the well-established HAR models.
Their findings highlight machine learning’s ability to capture long-memory effects and
incorporate additional predictors, yielding superior performance, especially for long-term
forecasts. Similarly, Dudek et al. expand this perspective by conducting a comparative
study of twelve forecasting methods, including GARCH, HAR, and various machine learning algorithms, across multiple cryptocurrencies. Their results emphasize the absence of
a universally superior model, with performance varying based on the error metric, forecast horizon, and specific cryptocurrency. Lastly, Mashrur et al. provide a broader view,
surveying the applications of machine learning in financial risk management, with a particular focus on its role in improving predictive accuracy and addressing challenges like
high-dimensional and noisy data.


Building on insights from the discussed papers and other relevant literature, you will
design group projects that apply the forecasting methods explored in the course to digital
currencies of their choice. The projects will emphasize practical application and critical
evaluation of statistical and machine learning techniques, allowing students to deepen
their understanding of volatility forecasting in dynamic cryptocurrency markets. You can
learn how to download historical crypto currency data from here: `[https://medium](https://medium.com/coinmonks/how-to-get-historical-crypto-currency-data-954062d40d2d)` _._ `com/`
`[coinmonks/how-to-get-historical-crypto-currency-data-954062d40d2d](https://medium.com/coinmonks/how-to-get-historical-crypto-currency-data-954062d40d2d)` . [1]

## **1.2 Stablecoin Depegging Analysis: May 2022 Stablecoin Crash**


Stablecoins have gone mainstream in the year of 2025. Stablecoins now rival the world’s
largest payment networks in transaction volume compared to Paypal and VISA. Stablecoin
transaction volume for the last 12 months was $9 trillion by September 2025. Despite this,
Crypto-asset prices collapsed in May 2022 when an algorithmic stablecoin, TerraUSD
(UST), failed to maintain its $1 price target (7 May 2022) and triggered widespread selloffs across the crypto space. Yet some stablecoins experienced notably less redemption
pressure than others.


Ronald Yip. (2022) reviewed the May 2022 Stablecoin Depegging event with an investigation on leading stablecoins and backgrounds of the Cryptocurrency ecosystem.
Invented to provide price stability in the volatile cryptocurrency market, stablecoins are
designed to have stable value, typically by being pegged to a fiat currency like the U.S.
dollar. In recent years, however, stablecoins have frequently deviated from their pegged
value and pose serious challenges to market participants. There is an urgent need for
timely and effective modelling of stablecoin depegging.


1 This is available for paid subscribers. If you need this article, please reach out to the instructors.


2


In the short term, stablecoin depegging analysis can help in early response to crypto
asset investment decisions; in the long term, it aids in understanding and monitoring the
flaws or vulnerabilities in the issuance and operations of stablecoins.


[• Ronald Yip. (2022). An event study on the may 2022 stablecoin market crash.](https://www.hkma.gov.hk/media/eng/publication-and-research/research/research-memorandums/2022/RM09-2022.pdf.) _Hong_

.
_Kong Monetary Authority_

## **Relevant Methodologies.**


Random Forest (Breiman, L., 2001) is a powerful ensemble learning method that can be
used for both classification and regression tasks. It builds on bagging (bootstrap aggregation) and constructs a multitude of decision trees “voting” together to make decisions

- predicting the categories or predicting the numbers. When provided with manually
created lag variables, Random Forest can be used in time series analysis and forecasting
and under certain circumstances shows strengths over Time Series Forecasting models
that essentially predict future values based on historical patterns. Y. H. Lee et al. (2025)
conducted a comprehensive analysis of stablecoin depegging risk with variables including
sentiment indicators. In terms of early warnings of stablecoin depegs, this paper however
has its limitation in practical use.


  - Breiman, L. (2001). Random forests. _Machine learning_, 45(1), 5-32.


  - Lee, Y. H., Chiu, Y. F., & Hsieh, M. H. (2025). Stablecoin depegging risk prediction.
_Pacific-Basin Finance Journal_, 90, 102640.


Principal Component Analysis (PCA) is a multivariate technique that analyzes a data
table in which observations are described by several inter-correlated variables. PCA’s goal
is to extract the important information from the data table, and represent it as a set of new
uncorrelated variables called principal components (PCs), thus removing multicollinearity
in variables. Principal components are a few linear combinations of the original variables
that maximally explain the variability of all the variables, see Greenacre et al (2022), Bro
& Smilde (2014). PCA can be generalized as factor analysis to handle heterogeneous sets
of variables, hence shows potentials in identifying latent risk factors as early warnings of
stablecoin depegs together with instability measures e.g. rolling-window volatility.


  - Greenacre, M., Groenen, P. J., Hastie, T., d’Enza, A. I., Markos, A., & Tuzhilina, E.
(2022). Principal component analysis. _Nature Reviews Methods Primers_, 2(1), 100.


  - Bro, R., & Smilde, A. K. (2014). Principal component analysis. _Analytical methods_,
6(9), 2812-2831.

## **Data.**


  - Stablecoin ERC20 Transactions Dataset (token_transfers.csv is optional)


.
[ERC20-stablecoins.zip](https://snap.stanford.edu/data/ERC20-stablecoins.html)


3


## **Suggested Tasks.**


  - Visualize and analyze the price data & news data, and their relationships in the
stablecoin markets during the pre-crash, crash, post-crash period.


  - Create and identify measures as early warnings of stablecoin depegs.


  - Predict the stablecoin prices for post-crash period.


  - Your own research question(s).

# **2 FinTech**

## **2.1 Empirical Asset Pricing via Machine Learning**


Predicting asset prices is a central research topic in finance due to its significant implications for investment strategies, risk management, and market efficiency. Accurate price
predictions allow market participants to make informed decisions, potentially leading to
higher returns, reduced risks, and more stable financial markets. However, forecasting
asset prices is a complex challenge, given the multifaceted nature of financial markets.
Factors such as market volatility, noise, external influences, and investor behavior make
traditional price prediction methods often inadequate, particularly when dealing with
high-dimensional, non-linear, and rapidly changing data.
In recent years, machine learning (ML) has emerged as a powerful tool to address
these challenges. ML methods offer advanced techniques capable of modeling complex
relationships, adapting to new data, and incorporating both structured and unstructured
information. As a result, machine learning has become an indispensable tool in modern
financial research.
To explore how ML method can be used to predict asset price, you can begin by reading
the following paper.


[• Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning.](https://doi.org/10.1093/rfs/hhaa009) _The_
_Review of Financial Studies_ 33(5), 2223-2273.
Package: `https://dachxiu` _._ `[chicagobooth](https://dachxiu.chicagobooth.edu/)` _._ `edu/`


Gu et al. explores the application of machine learning (ML) methods to predict equity
risk premiums, demonstrating their superiority over traditional regression-based models. By effectively handling high-dimensional data and capturing complex non-linear
relationships, ML methods, such as regression trees and neural networks, achieve better
predictive accuracy and deliver substantial economic gains, including improved portfolio
Sharpe ratios. The study also identifies key predictors like momentum, liquidity, and
volatility while establishing ML as a powerful tool for advancing empirical asset pricing
frameworks and practical financial strategies.


Inspired by the seminar work by Gu et al. there are a number of following up researches, including:


4


  - Leippold, M., Wang, Q., & Zhou, W. (2022). Machine learning in the Chinese stock
market. _Journal of Financial Economics_ 145(2), 64-82.


  - Drobetz, W., & Otto, T. (2021). Empirical asset pricing via machine learning: evidence from the European stock market. _Journal of Asset Management_ 22(7), 507-538.


  - Cakici, N., & Zaremba, A. (2022). Empirical asset pricing via machine learning: The
global edition. _Available at SSRN_ 4028525.


Based on the insights from these and other relevant papers, you will design group
projects that apply machine learning methods in asset price prediction. Some possible
research topics include:


  - look at a different equity market (e.g. Singapore)


  - explore different assets (e.g. bonds, cryptocurrency)


  - apply different ML method (e.g. recurrent neural networks)


Please note the you are not restricted in the above topics, do feel free to design your own
topic. Just to make sure that you get the data needed for your topic.

## **2.2 Machine-Learning Skills for Mutual Fund**


Mutual funds are a cornerstone of modern financial markets, managing trillions of dollars
and offering accessible investment opportunities for individuals and institutions. By the
end of 2024, U.S. mutual funds managed over $29 trillion, playing a crucial role in retirement savings and wealth accumulation. Despite their prominence, a longstanding debate
persists about whether actively managed funds can consistently outperform benchmarks.
While some fund managers demonstrate skill in generating net-of-fees abnormal returns,
identifying these outperformers remains a significant challenge.


In response to this challenge, machine learning (ML) has emerged as a transformative
tool in financial research over the past decade. Unlike traditional econometric methods, which often struggle to capture the intricate, non-linear relationships in financial
data, machine learning excels at processing large datasets and uncovering hidden patterns. Advanced algorithms, such as neural networks, have proven particularly effective
in predicting fund performance, offering new opportunities to distinguish skilled fund
managers and enhance decision-making in asset management.


To explore how ML method can be used to predict mutual fund performance, you can
begin by reading the following papers.


[• Kaniel, R., Lin, Z., Pelger, M., & Van Nieuwerburgh, S. (2023). Machine-learning the](https://www-sciencedirect-com.libproxy1.nus.edu.sg/science/article/pii/S0304405X23001253)
[skill of mutual fund managers.](https://www-sciencedirect-com.libproxy1.nus.edu.sg/science/article/pii/S0304405X23001253) _Journal of Financial Economics_ 150(1), 94-138.
Package: `https://data` _._ `mendeley` _._ `[com/datasets/jxzgfw47wg/1](https://data.mendeley.com/datasets/jxzgfw47wg/1)`


5


[• DeMiguel, V., Gil-Bazo, J., Nogales, F. J., & Santos, A. A. (2023). Machine learning](https://www.sciencedirect.com/science/article/pii/S0304405X23001770)
[and fund characteristics help to select mutual funds with positive alpha.](https://www.sciencedirect.com/science/article/pii/S0304405X23001770) _Journal of_
_Financial Economics_ 150(3), 103737.
Package: `https://data` _._ `mendeley` _._ `[com/datasets/rpgb99m5zy/3](https://data.mendeley.com/datasets/rpgb99m5zy/3)`


Kaniel et al. uses feed forward neural network to identify key predictors of mutual fund
performance, focusing on net-of-fees abnormal returns. DeMiguel et al. demonstrates
that machine learning methods, particularly gradient boosting and random forests, can
effectively predict and select actively managed mutual funds with significant positive net
alphas.
Based on the insights from these and other relevant papers, you will design group
projects. Some possible research topics include:


  - look at a different mutual fund market or different type of mutual fund (e.g. bond)


  - explore sector-specific prediction of mutual fund (e.g. healthcare fund)


  - apply different ML method (e.g. recurrent neural networks)


Please note the you are not restricted in the above topics, do feel free to design your own
topic. Just to make sure that you get the data needed for your topic.

# **3 Digital Economy**

## **3.1 Causal Forest**


Causal forests, a machine learning method designed for estimating heterogeneous treatment effects, have become a cornerstone in modern data analysis. Their ability to identify
and analyze the impact of various interventions across diverse contexts is particularly vital
in the digital economy, where businesses and policymakers often rely on nuanced insights
to drive decisions. For instance, in targeted advertising, causal forests help determine the
effectiveness of promotional campaigns by estimating the differential impact on distinct
user segments. This insight enables companies to allocate resources more effectively, ensuring advertisements resonate with the intended audience. Similarly, by analyzing user
responses to new platform features, companies can determine which demographic groups
benefit most from specific changes, thereby refining their product development strategies.


To equip students with the tools to explore these opportunities, foundational research
in causal inference provides a vital starting point. The seminal work of Susan Athey, Guido
Imbens, and their collaborators bridges the gap between theoretical rigor and practical
application:


[• Athey, S., & Imbens, G. (2016). “Recursive partitioning for heterogeneous causal](https://doi.org/10.1073/pnas.1510489113)
[effects.”](https://doi.org/10.1073/pnas.1510489113) _Proceedings of the National Academy of Sciences_ 113(27): 7353-7360.
Package: `https://github` _._ `[com/susanathey/causalTree](https://github.com/susanathey/causalTree)`


6


[• Athey, S., Tibshirani J., and Wager S. (2019): “Generalized random forests.”](https://doi.org/10.1214/18-AOS1709) _Annals_
_of Statistics_, 47(2): 1148-1178.
Package: `https://grf-labs` _._ `[github](https://grf-labs.github.io/grf/index.html)` _._ `io/grf/index` _._ `html`


[• Wager, S., & Athey, S. (2018): “Estimation and inference of heterogeneous treatment](https://doi.org/10.1080/01621459.2017.1319839)
[effects using random forests.”](https://doi.org/10.1080/01621459.2017.1319839) _Journal of the American Statistical Association_ 113(523):
1228-1242.


Athey and Imbens (2016) introduces causal trees, a method that uses recursive partitioning to estimate treatment effects across subgroups. This approach allows researchers to
identify nuanced patterns of heterogeneity while maintaining statistical validity through
techniques like cross-validation. Building on this foundation, Wager and Athey (2018)
extend the methodology to causal forests. This paper provides a comprehensive framework for estimating heterogeneous treatment effects using nonparametric methods. It
establishes the consistency and asymptotic normality of causal forests, offering practical approaches for constructing confidence intervals. They also demonstrate how causal
forests outperform traditional techniques, particularly in high-dimensional data settings,
solidifying their relevance in fields like marketing, policy evaluation, and healthcare.


Building on insights from these and other relevant papers, you will design group
projects that apply the discussed methods to practical settings. One option is to find an
existing experiment dataset and identify which subpopulation benefit the most from the
intervention. Below are some potential applications:


  - Athey, S., & Wager S. (2019). Estimating treatment effects with causal forests: An
application. _Observational studies_ 5(2): 37-51.
Data: `https://github` _._ `[com/grf-labs/grf/tree/master/experiments/acic18](https://github.com/grf-labs/grf/tree/master/experiments/acic18)`


  - Britto, D. G., Pinotti P., & Sampaio B. (2022). The effect of job loss and unemployment
insurance on crime in Brazil. _Econometrica_ 90(4): 1393-1423.
Data: `https://www` _._ `econometricsociety` _._ `[org/publications/econometrica/2022/](https://www.econometricsociety.org/publications/econometrica/2022/07/01/effect-job-loss-and-unemployment-insurance-crime-brazil)`
```
  07/01/effect-job-loss-and-unemployment-insurance-crime-brazil

```

  - Davis, Jonathan MV, and Sara B. Heller. “Using causal forests to predict treatment
heterogeneity: An application to summer jobs.” _American Economic Review_ 107, no.
5 (2017): 546-550.


[• Huddy, L., Mason, L., & Aarøe, L. (2015). Expressive partisanship: Campaign in-](https://doi.org/10.1017/S0003055414000604)
[volvement, political emotion, and partisan identity.](https://doi.org/10.1017/S0003055414000604) _American Political Science Review_,
109(1), 1-17.


  - Klüver, H., Hartmann, F., Humphreys, M., Geissler, F., & Giesecke, J. (2021). Incentives can spur COVID-19 vaccination uptake. Proceedings of the National Academy
of Sciences, 118(36), e2109543118.
Data: `https://github` _._ `[com/wzb-ipi/covid_hesitancy_2021](https://github.com/wzb-ipi/covid_hesitancy_2021)`


7


[• Oreopoulos, P. (2011). Why do skilled immigrants struggle in the labor market? A](https://www.aeaweb.org/articles?id=10.1257/pol.3.4.148)
[field experiment with thirteen thousand resumes.](https://www.aeaweb.org/articles?id=10.1257/pol.3.4.148) _American Economic Journal: Eco-_
_nomic Policy_, 3(4), 148-171.
Data: `https://www` _._ `openicpsr` _._ `[org/openicpsr/project/114770/version/V1/view](https://www.openicpsr.org/openicpsr/project/114770/version/V1/view)`

## **3.2 Public Tranasport Resilience during Major Events**


Large-scale public events often impact urban transport systems, leading to distruptions
in normal operations, increased congestion, longer travel times, and welfare losses for
commuters. This project draws on empirical evidence from Macau’s Grand Prix (Kim and
Wang, 2025) to study how major events affect public transport performance.
By analyzing changes in travel time, traffic conditions, and passenger crowding before,
during, and after the event, students will quantify the impact of large-scale events on
public transport systems and derive insights that are relevant to Singapore’s context. This
study aims to identify factors and strategies that enhance transport resilience and inform
policy makers on how to better manage urban mobility during major events. Below are
some related data sets and literature:


  - Kim, S. K., & Wang, J. (2025). A dataset on public bus transportation during normal
and Grand Prix Seasons in the Macao area. _Scientific Data_, 12(1), 1306.
Data: `https://github` _._ `[com/amangkim/MOPubBus/](https://github.com/amangkim/MOPubBus/)`


  - Anderson, M. L. (2014). Subways, strikes, and slowdowns: The impacts of public
transit on traffic congestion. _American Economic Review_, 104(9), 2763-2796.


  - Yang, J., Purevjav, A. O., & Li, S. (2020). The marginal cost of traffic congestion and
road pricing: Evidence from a natural experiment in Beĳing. _American Economic_
_Journal: Economic Policy_, 12(1), 418-453.


Based on insights from these and related studies, you will design a group project on
public transport resilience during major events in Singapore (e.g., the Singapore Grand
Prix, National Day Parade). Your project should go beyond descriptive analysis. Some
possible directions include:


  - Quantify changes in public transport performance before, during, and after major
events. Assess the duration and persistence of disruptions.


  - Identify and measure spillover effects of major events on non-event areas, alternative
routes, or other transport modes. Assess the effectiveness of mitigation measures.


  - Develop and compare empirical or machine learning methods to predict congestion,
delays, or crowding during major events.


Note that you are not restricted to the directions listed above and are encouraged to
propose your own research question. Just to make sure that you get the data needed for
your topic.


8


