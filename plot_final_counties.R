rm(list=ls())

library(data.table)
library(rgdal)
library(maptools)
library(maps)

fips <- read.csv("data/fips_list.txt",header=FALSE,colClasses="character")
fips <- fips$V1
#stns <- get(load("data/list_stations_by_county_final.Rdata"))

demo <- read.csv("data/demo.csv",header=TRUE)
ix1 <- nchar(demo$FIPS)==4
demo$FIPS[ix1] <- paste0("0",demo$FIPS[ix1])
demo <- subset(demo, FIPS %in% fips)

shps <- readOGR("notebooks/cb_2018_us_county_20m","cb_2018_us_county_20m")
shps$FIPS <- paste0(shps$STATEFP,shps$COUNTYFP)
#shps$modelled <- rep("no",nrow(shps))
cases <- read.csv("data/dailycases.csv",header=TRUE)
cases$X <- NULL
ix1 <- nchar(cases$FIPS)==4
cases$FIPS[ix1] <- paste0("0",cases$FIPS[ix1])
cases <- as.data.table(cases)
cases1 <- copy(cases)
cases1$FIPS <- NULL
cases1[,accum:=rowSums(.SD)]
cases1$FIPS <- cases$FIPS
cases1 < - subset(cases1,FIPS %in% fips)

shps$accum <- NA
shps$accum <- cases1$accum[match(shps$FIPS,cases$FIPS)]
shps$popdens <- demo$Pop..Dens[match(shps$FIPS,demo$FIPS)]
shps$accum1 <- shps$accum / shps$popdens
shps$accum1[shps$accum1>50] <- 50

# use US state boundary as background
usmap <- map(database="state",plot=FALSE,resolution=0)
us <- map2SpatialLines(usmap,proj4string=CRS("+proj=longlat +a=6370000 +b=6370000"))
panel.layout <- list(list("sp.lines", spTransform(us,shps@proj4string), lwd=1.3, col=1))

outFile <- "figs/counties_cases.png"
if(!dir.exists(dirname(outFile))) dir.create(dirname(outFile),recursive=TRUE)

png(outFile,width=7.5,height=5,units="in",res=300)
print(spplot(shps,zcol="accum1", sp.layout=panel.layout, xlim=c(-125, -66), ylim=c(24.5, 50), scales=list(draw=TRUE), lwd=0.5,#col=NA,
 main="Cumulative cases for U.S. counties normalized by population density"))
dev.off()

shps$pov <- demo$X..Poverty[match(shps$FIPS,demo$FIPS)]
outFile <- "figs/counties_pov.png"
png(outFile,width=7.5,height=5,units="in",res=300)
print(spplot(shps,zcol="pov", sp.layout=panel.layout, xlim=c(-125, -66), ylim=c(24.5, 50), scales=list(draw=TRUE), lwd=0.5,#col=NA,
             main="% Poverty for U.S. counties"))
dev.off()

shps$wht <- demo$X..White[match(shps$FIPS,demo$FIPS)]
shps$wht[shps$wht>1] <- 1
outFile <- "figs/counties_wht.png"
png(outFile,width=7.5,height=5,units="in",res=300)
print(spplot(shps,zcol="wht", sp.layout=panel.layout, xlim=c(-125, -66), ylim=c(24.5, 50), scales=list(draw=TRUE), lwd=0.5,#col=NA,
             main="% White population in U.S. counties"))
dev.off()

shps$nonwht <- demo$X..Non.white[match(shps$FIPS,demo$FIPS)]
shps$nonwht[shps$nonwht>0.5] <- 0.5
outFile <- "figs/counties_nonwht.png"
png(outFile,width=7.5,height=5,units="in",res=300)
print(spplot(shps,zcol="nonwht", sp.layout=panel.layout, xlim=c(-125, -66), ylim=c(24.5, 50), scales=list(draw=TRUE), lwd=0.5,#col=NA,
             main="% Non-White population in U.S. counties"))
dev.off()
