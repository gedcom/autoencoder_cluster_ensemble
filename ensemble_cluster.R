library("tidyverse")
library("mclust")
library('beeswarm')
library("magrittr")
library("clue")
library("parallel")
library("SIMLR")

numCores = 8

arglist = commandArgs(trailingOnly = TRUE)

argtypes = list(
    data.frame(long = "input",  short = "i", type = "character", default = NA, stringsAsFactors = FALSE),
    data.frame(long = "output", short = "o", type = "character", default = NA, stringsAsFactors = FALSE),
    data.frame(long = "labels", short = "l", type = "character", default = NA, stringsAsFactors = FALSE),
    data.frame(long = "ensemblesize", short = "e", type = "numeric", default = 20, stringsAsFactors = FALSE)
) %>% do.call(rbind, .)

argtbl = lapply(seq_along(arglist), function(i) {
    arg = arglist[[i]]
    shorts = c()
    longs = c()
    # Get option names
    if (substring(arg, 1, 2) == "--") {
        longs = substring(arg, 3)   
    } else if (substring(arg, 1, 1) == "-") {
        shorts = substring(arg, 2) %>% strsplit(split = "", fixed = TRUE) %>% unlist
        longs = sapply(shorts, function(short) {
            if (short %in% argtypes$short) {
                return(argtypes$long[argtypes$short == short])
            } else return(NA)
        })
    } else return(NULL)
    
    # Get values for options
    for (j in seq_along(longs)) {
        long = longs[[j]]
        val = arglist[[i + j]]
        return(data.frame(option = long, value = val, stringsAsFactors = FALSE))
        
    }
}) %>% do.call(rbind, .)

req.opts = argtypes$long

# Check all required options have been specified
lapply(req.opts, function(option) {
    if (!(option %in% argtbl$option)) {
        long = paste0("--", option)
        short = paste0("-", argtypes$short[argtypes$long == option])
        errormessage = paste("Option missing:", long)
        stop(errormessage)
    }
})

inputdir = argtbl$value[argtbl$option == "input"]
outputpath = argtbl$value[argtbl$option == "output"]
labelpath = argtbl$value[argtbl$option == "labels"]
ensemble.range = argtbl$value[argtbl$option == "ensemblesize"]  %>% strsplit(split = ",", fixed = TRUE) %>% unlist %>% as.numeric

# Evaluation metrics
getEvaluation <- function(cluster, label) {
    accdf <- data.frame(
        ARI = mclust::adjustedRandIndex(as.numeric(factor(label)), as.numeric(factor(cluster))), # ARI
        NMI = igraph::compare(as.numeric(factor(cluster)), as.numeric(factor(label)), method = "nmi"), # NMI
        FM = dendextend::FM_index(as.numeric(factor(label)), as.numeric(factor(cluster))), # FM
        Jaccard = clusteval::cluster_similarity(as.numeric(factor(label)), as.numeric(factor(cluster)), 
                                      similarity = "jaccard", method = "independence") # Jaccard
    )
    return(accdf)
}

rept <- function(seed, path, estimate.k = TRUE, size, truelbs){

    if (estimate.k) cat("Estimating k\n") else cat("Getting k from ground truth\n")
    
    fs <- list.files(path, pattern = "^\\d+.csv")
    n <- ifelse(length(fs) < size, length(fs), size)
    k = length(unique(truelbs))    

    fs[1:n] %>% paste(path, . , sep="/") -> fps
    cat("Beginning parallel clustering...\n")
    
    savelist = list.files(path = path, pattern = paste0(seed, "_.*", "rds"))
    numsaves = length(savelist)
    cat(numsaves, "saved clusterings found.\n")
    
    mcmapply(function(fp, i, seed, k) {
        if (is.na(fp)) {        
            if (numsaves < n) {
                sleeptime = i*120
                cat("Node ", i, " sleeping for ", sleeptime, " seconds.\n") 
                Sys.sleep(sleeptime)
                cat("Node ", i, " ready to go.\n")
            }
            return(NULL)
        } else {

            tempfn = paste0(path, "/", seed, "_", i, ".rds")
            if (file.exists(tempfn)) {
                out = readRDS(tempfn)
                cat("SIMLR output for replicate ", seed, ", dataset ", i, " reloaded from previous run.\n")
                return(out)
            }
            
            cat("Loading dataset ", i, "... ", sep = "")
            fp %>% read.csv2(sep = ",", header = FALSE) %>% apply(2, as.numeric) %>% t -> dat
            cat("Finished loading. Clustering dataset...\n")

            set.seed(seed*i^2)
            
            # Use SIMLR for clustering
            out = SIMLR::SIMLR(dat, c = k, cores.ratio = 0.5)$y
            
            # Use k-means for clustering
            #out = kmeans(t(dat), centers = k)
            
            cat("Finished processing dataset ", i, ". Backing up output.\n")
            saveRDS(out, file = tempfn)
            return(out)
        }
    }, c(rep(NA, min(numCores, n) - 1), fps), c(seq(min(numCores, n) - 1), seq(n)), MoreArgs = list(seed = seed, k = k), SIMPLIFY = FALSE, mc.cores = numCores, mc.preschedule = FALSE) -> clusts
    clusts = clusts[numCores:length(clusts)]
    
    cat("Ensemble completed.\n")
    
    return(clusts)
}

compare = function(path, truelbs, estimate.k, range=c(1, 5, 10, 20, 50, 100)) {
    lapply(1:10, function(i) {
        cat("Test: ", i, "Max ensemble size: ", max(range), "\n")
        allclusts = rept(seed=i, path=path, estimate.k = estimate.k, size=max(range), truelbs=truelbs)
        lapply(range, function(j) {
            if (j == 1) {
                cat("Getting single clustering accuracy...\n")
                getEvaluation(allclusts[[1]]$cluster, truelbs) %>% mutate(n = 1) %>% return()
            } else {
                clusts = allclusts[1:j]
                
                # map multiple clustering
                consensus = cl_consensus(clusts, method = "HE")
                
                cat("Getting ensemble clustering accuracy for size ", j, "...\n")
                as.matrix(consensus[[1]]) %>% apply(1, function(x) which(x == 1)) %>% as.character() %>%
                    getEvaluation(truelbs) %>% mutate(n = j) %>% return()
            }
        }) %>% bind_rows
    }) %>% bind_rows %>% arrange(n) %>% select(n, ARI, NMI, FM, Jaccard) %>% return()
}

## Data1
cat("Loading labels from ", labelpath, "...\n")
truelbs = as.character(read.csv2(labelpath, sep=",", header = TRUE)[,1])
cat("Labels loaded.\n")

dirlist = list.dirs(inputdir, full.names = TRUE, recursive = FALSE)
hpkey = list(
    "l" = "learning_rate",
    "a" = "ae_dim",
    "p" = "random_proj_dim"
)

# get name of directory from path
inputdirname = strsplit(inputdir, split = "_", fixed = TRUE)

resultsdf = lapply(dirlist, function(batchdir) {
    cat("Processing ", batchdir, ".\n")
    # Extract hyperparameter values from filename
    
    hpstrings = batchdir %>%
        substring(first = nchar(inputdirname) + 2) %>%
        strsplit(split = "_", fixed = TRUE) %>% extract2(1)
    
    dfvals = substring(hpstrings, first = 2) %>% as.numeric
    names(dfvals) = substring(hpstrings, first = 1, last = 1) %>% sapply(function(x) hpkey[[x]])
    dfvals = dfvals[order(names(dfvals))] %>% na.omit %>% as.list %>% as.data.frame
    
    # Prepare data frame containing accuracies for gold-standard k
    ensemble = compare(path = batchdir, estimate.k = FALSE, truelbs = truelbs, range = ensemble.range)
    
    cbind(dfvals, ensemble) %>% return()
}) %>% do.call(rbind, .)

# Write process results to a file.
cat("Done.")

write.csv(resultsdf, file = outputpath, row.names = FALSE, col.names = TRUE)


