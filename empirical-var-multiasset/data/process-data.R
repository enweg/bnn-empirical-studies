library(tidyverse)

df <- read_csv("./oxfordmanrealizedvolatilityindices-3.csv")
colnames(df)[1] <- "Date"
df <- df %>%
  select(Date, Symbol, rv5, close_price) %>%
  group_by(Symbol) %>%
  arrange(Date) %>%
  # Calculating return as log differences
  mutate(return = log(close_price) - log(lag(close_price)),
         returnsq = return^2, 
         Symbol = str_remove_all(Symbol, "\\.")) %>%
  ungroup()

# Keeping only Symbols that were already available from 
# February 2000 on
keep_symbols <- df %>%
  group_by(Symbol) %>%
  summarise(min_date = min(Date), 
            .groups = "drop") %>%
  filter(min_date <= as.Date("2000/02/01")) %>%
  pull(Symbol)
df <- df %>%
  filter(Symbol %in% keep_symbols)

# Bringing the data into wide format. 
# This will be required at a later point.
df.wide <- df %>%
  select(-close_price) %>%
  pivot_wider(id_cols = Date, 
              names_from = Symbol, 
              values_from = c(rv5, return, returnsq), names_sep = "_")

# Bringing the data into wide format also has the advantage 
# of showing as all the dates for which at least one asset had 
# an observation. Assets that did not have an observation on a certain
# day will have a missing value. We want to keep all those assets that
# had at last as many non-missing observations as SPX
keep_variables <- df.wide %>%
  summarise(across(.fns = function(x) sum(is.na(x)))) %>%
  pivot_longer(cols = everything(), 
               names_to = "Variable", 
               values_to = "n.na") %>%
  # filter(Variable == "return.SPX")
  filter(n.na <= 260) %>%
  pull(Variable)
df.wide <- df.wide[keep_variables] %>%
  drop_na()

write_csv(df.wide, "./wide-data.csv")

# These are the Symbols that survived all cleaning
kept_symbols <- colnames(df.wide)
kept_symbols <- kept_symbols[str_detect(kept_symbols, "rv5|return")]
kept_symbols <- str_match_all(kept_symbols, "_([A-Za-z0-9]+)")
kept_symbols <- sort(unique(do.call(rbind, kept_symbols)[, 2]))
tibble(Symbol = kept_symbols) %>%
  write_csv("./kept-symbols.csv")


