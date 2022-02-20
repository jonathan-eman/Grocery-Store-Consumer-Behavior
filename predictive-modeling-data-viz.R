library(tidyverse)
setwd("~/Fall 2021 Courses/COMM 4351")

data_raw <- read.csv("marketing_campaign.csv", sep="\t")

data_raw %>%
  mutate(
    Is_Parent = case_when(
      (Kidhome > 0 | Teenhome > 0) ~ "Parent",
      (Kidhome == 0 & Teenhome == 0) ~ "Non-Parent" 
    ),
    total_purchases = NumWebPurchases + NumCatalogPurchases + NumStorePurchases,
    total_spend = MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts,
    avg_purchase_size = total_spend / total_purchases,
    discount_prop = NumDealsPurchases / total_purchases,
    cat_prop = NumCatalogPurchases / total_purchases
  ) %>%
  filter(total_purchases > 0) -> data

data %>%
  group_by(Is_Parent) %>%
  summarize(
    wine = mean(MntWines),
    meat = mean(MntMeatProducts),
    sweets = mean(MntSweetProducts),
    fish = mean(MntFishProducts),
    fruit = mean(MntFruits)
  ) -> total_spend

data %>%
  group_by(Is_Parent) %>%
  summarize(
    avg_purchase_size = mean(avg_purchase_size),
    avg_num_purchases = mean(total_purchases),
    avg_total_spend = mean(total_spend),
    freq = mean(104/total_purchases)
  ) -> rfm

total_spend %>%
  gather(key="category", value = "total_spend", -Is_Parent) %>%
  ggplot(aes(x=category, y=total_spend)) + geom_col(fill='purple') + facet_wrap(~Is_Parent) +
  theme(panel.background = element_rect(fill = "white", colour = "grey50")) +
  labs(x="Category", y="Avg. Total Spend (past 2 years)") + 
  geom_text(aes(label=round(total_spend, 2)), vjust=-.3, size=5) +
  theme(axis.text=element_text(size=16),
        axis.title=element_text(size=16),
        strip.text.x = element_text(size = 16))

data %>%
  ggplot(aes(NumDealsPurchases)) + geom_histogram(binwidth = 2, fill='purple', color='black') +
  facet_wrap(~Is_Parent) +
  theme(panel.background = element_rect(fill = "white", colour = "grey50")) +
  labs(x="# of Purchases with Coupons", y="# of Customers") +
  theme(axis.text=element_text(size=16),
        axis.title=element_text(size=16),
        strip.text.x = element_text(size = 16))

data %>%
  ggplot(aes(NumCatalogPurchases)) + geom_histogram(binwidth = 2, fill='purple', color='black') +
  facet_wrap(~Is_Parent) +
  theme(panel.background = element_rect(fill = "white", colour = "grey50")) +
  labs(x="# of Purchases from Catalog", y="# of Customers") +
  theme(axis.text=element_text(size=16),
        axis.title=element_text(size=16),
        strip.text.x = element_text(size = 16))

rfm %>%
  ggplot(aes(x=Is_Parent, y=avg_purchase_size)) + geom_col(fill='purple') +
  theme(panel.background = element_rect(fill = "white", colour = "grey50")) +
  labs(x="Parent Status", y="Avg. Purchase Size ($)") + 
  geom_text(aes(label=round(avg_purchase_size, 2)), vjust=-.3, size=7) +
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=20),
        strip.text.x = element_text(size = 20))

rfm %>%
  ggplot(aes(x=Is_Parent, y=freq)) + geom_col(fill='purple') +
  theme(panel.background = element_rect(fill = "white", colour = "grey50")) +
  labs(x="Parent Status", y="Weeks per Purchase") + 
  geom_text(aes(label=round(freq, 2)), vjust=-.3, size=7) +
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=20),
        strip.text.x = element_text(size = 20))


