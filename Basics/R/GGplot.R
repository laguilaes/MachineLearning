library(ggplot2)
# https://ggplot2.tidyverse.org/reference/index.html



# Cargar datos------------------------------------------------------------------
data("midwest", package = "ggplot2")

#Multiples graficas-------------------------------------------------------------
library(gridExtra)
grid.arrange(plot1, plot2, ncol=2)

# X11---------------------------------------------------------------------------
x11(width = 500, height = 300)

#Temas GGPlot-------------------------------------------------------------------
library(ggthemes)

#Setear el tema
theme_set(
  theme_wsj() +
    theme(
      axis.title.x=element_text(vjust=10, size=15),  
      axis.title.y=element_text(size=15),  
      axis.text.x=element_text(size=10, angle = 30, vjust=.5), 
      axis.text.y=element_text(size=10),
      axis.line.x = element_line(colour = "darkorange", 
                                 size=1.5, 
                                 lineend = "butt"),
      axis.line.y = element_line(colour = "darkorange", 
                                 size=1.5),

      
      legend.title = element_text(size=12, color = "firebrick"), 
      legend.text = element_text(size=10),
      legend.key=element_rect(fill='springgreen'),
      legend.position = "None",
      
      panel.background = element_rect(fill = 'steelblue'),
      panel.grid.major = element_line(colour = "firebrick", size=3),
      panel.grid.minor = element_line(colour = "blue", size=1),
      panel.border = element_blank(),
      
      plot.background=element_rect(fill="steelblue"), 
      plot.margin = unit(c(2, 4, 1, 3), "cm"),
      plot.title=element_text(size=20, 
                              face="bold", 
                              family="American Typewriter",
                              color="tomato",
                              hjust=0.5,
                              lineheight=1.2), 
      plot.subtitle=element_text(size=15, 
                                 family="American Typewriter",
                                 face="bold",
                                 hjust=0.5), 
      plot.caption=element_text(size=15),
    )
)

#temas base
theme_gray()
theme_bw()
theme_linedraw()
theme_light()
theme_minimal()
theme_classic()
theme_void()



#Ejemplos----------------------------------------------------------------------

# Meter el aes dentro de ggplot afecta a todos los geom. 
ggplot(diamonds, ) + 
  geom_point(aes(x = carat, y = price, color = cut)) +
  geom_smooth(aes(x = carat, y = price)) + 
  facet_wrap(~cut, scales="free") +   #una o varias variables
  scale_color_discrete(name="Cut of diamonds") + #scale_shape_discrete, o scale_shape_continuous
  ggtitle("Diamonds")+
  labs(title="Scatterplot", x="Carat", y="Price") + 
  theme(plot.title=element_text(size=30, face="bold"), 
        axis.text.x=element_text(size=15), 
        axis.text.y=element_text(size=15),
        axis.title.x=element_text(size=25),
        axis.title.y=element_text(size=25)) 




ggplot(midwest, aes(x=area, y=poptotal)) + 
  geom_point() + 
  geom_smooth(method="lm") + 
  coord_cartesian(xlim=c(0,0.1), ylim=c(0, 500000)) 




ggplot(midwest, aes(x=area, y=poptotal)) + 
  geom_point(aes(col=state), size=3) + 
  geom_smooth(method="lm", col="firebrick") +
  scale_x_continuous(breaks=seq(0, 0.1, 0.01), labels = letters[1:11]) +
  scale_colour_brewer(palette = "Set1") +
  coord_cartesian(xlim=c(0, 0.1), ylim=c(0, 1000000)) + 
  labs(title="Area Vs Population", subtitle="From midwest dataset", y="Population", x="Area", caption="Midwest Demographics")





ggplot(midwest, aes(x=area, y=poptotal)) + 
  geom_point(aes(col=state, size=popdensity)) + 
  geom_smooth(method="loess", se=F) + xlim(c(0, 0.1)) + ylim(c(0, 500000)) + 
  labs(title="Area Vs Population", y="Population", x="Area", caption="Source: midwest") +
  scale_color_manual(name="State", 
                        labels = c("Illinois", 
                                   "Indiana", 
                                   "Michigan", 
                                   "Ohio", 
                                   "Wisconsin"), 
                        values = c("IL"="blue", 
                                   "IN"="red", 
                                   "MI"="green", 
                                   "OH"="brown", 
                                   "WI"="orange")) 
  

library(grid)
midwest_sub <- midwest[midwest$poptotal > 300000, ]
midwest_sub$large_county <- ifelse(midwest_sub$poptotal > 300000, midwest_sub$county, "")
ggplot(midwest, aes(x=area, y=poptotal)) + 
  geom_point(aes(col=state, size=popdensity)) + 
  geom_smooth(method="loess", se=F) + xlim(c(0, 0.1)) + ylim(c(0, 500000)) + 
  labs(title="Area Vs Population", y="Population", x="Area", caption="Source: midwest") +
  geom_text(aes(label=large_county), size=2, data=midwest_sub) + 
  geom_label(aes(label=large_county), size=2, data=midwest_sub, alpha=0.25) +
  annotation_custom(grid.text("Some text...", x=0.7,  y=0.8, gp=gpar(col="firebrick", fontsize=14, fontface="bold")))



ggplot(midwest, aes(x=area, y=poptotal)) + 
  geom_point(aes(col=state, size=popdensity)) + 
  geom_smooth(method="loess", se=F) + xlim(c(0, 0.1)) + ylim(c(0, 500000)) + 
  scale_x_reverse() + 
  scale_y_reverse()



ggplot(mpg, aes(x=displ, y=hwy)) + 
  geom_point() + 
  labs(title="hwy vs displ", caption = "Source: mpg", subtitle="Faceting") +
  geom_smooth(method="lm", se=FALSE) + 
  facet_grid(manufacturer ~ class)






############################### GEOMS #########################################
# Scatterplot-------------------------------------------------------------------


ggplot(midwest, aes(x=area, y=poptotal)) + 
  geom_point(aes(col=state, size=popdensity)) + 
  geom_smooth(method="loess", se=F) + 
  xlim(c(0, 0.1)) + 
  ylim(c(0, 500000)) + 
  labs(subtitle="Area Vs Population", 
       y="Population", 
       x="Area", 
       title="Scatterplot", 
       caption = "Source: midwest")

# Marginal----------------------------------------------------------------------
ggplot(mtcars, aes(wt, mpg)) +
  geom_point() + 
  geom_rug()

# Area--------------------------------------------------------------------------
huron <- data.frame(year = 1875:1972, level = as.vector(LakeHuron))
ggplot(huron, aes(year)) +
geom_ribbon(aes(ymin=0, ymax=level))
# Lines ------------------------------------------------------------------------
ggplot(economics, aes(date, unemploy)) + geom_line()
# Encyrcling-------------------------------------------------------------------
ggplot(faithful, aes(waiting, eruptions, color = eruptions > 3)) +
  geom_point() +
  stat_ellipse()
# Segment ----------------------------------------------------------------------
df <- data.frame(x1 = 2.62, x2 = 3.57, y1 = 21.0, y2 = 15.0)

 
ggplot(mtcars, aes(wt, mpg)) +
  geom_point() + 
  geom_curve(aes(x = x1, y = y1, xend = x2, yend = y2, colour = "curve"), data = df) +
  geom_segment(aes(x = x1, y = y1, xend = x2, yend = y2, colour = "segment"), data = df)

# Jitter------------------------------------------------------------------------

ggplot(mpg, aes(cty, hwy)) +
  geom_jitter(width = .5, size=1) 


# Count chart-------------------------------------------------------------------
ggplot(mpg, aes(cty, hwy)) +
  geom_count(col="tomato3", show.legend=F)



# Barcharts -------------------------------------------
ggplot(mpg, aes(y = class)) +
  geom_bar(aes(fill = drv), position = position_stack(reverse = TRUE)) +
  theme(legend.position = "top")


# Heatmap ----------------------------------------------------------------------

ggplot(faithfuld, aes(waiting, eruptions)) +
  geom_tile(aes(fill = density)) +
  scale_fill_distiller(palette = "Spectral")


# Contour ----------------------------------------------------------------------
ggplot(faithfuld, aes(waiting, eruptions, z = density)) +
  geom_contour()


# Polygon-----------------------------------------------------------------------
d=data.frame(x=c(1,2,2, 3,4,4), y=c(1,1,2, 2,2,3), t=c('a', 'a', 'a',  'b', 'b', 'b'), r=c(1,2,3, 4,5,6))
ggplot() +
  geom_polygon(data=d, mapping=aes(x=x, y=y, group=t))

# Histogram -------------------------------------------------------------------

ggplot(mpg, aes(displ)) + 
  scale_fill_brewer(palette = "Spectral") +
  geom_histogram(aes(fill=class), 
                   binwidth = .1, 
                   col="black", 
                   size=.1) 


# QQplots ---------------------------------------------------------------------
ggplot(mtcars, aes(sample = mpg)) +
  stat_qq() +
  stat_qq_line()
# Density ---------------------------------------------------------------------

ggplot(mpg, aes(displ)) + 
  scale_fill_brewer(palette = "Spectral") +
  geom_density(aes(fill=class), 
                 col="black", 
                 size=.1) 



# Density2D -------------------------------------------------------------------
ggplot(faithful, aes(x = eruptions, y = waiting)) +
  geom_point() +
  xlim(0.5, 6) +
  ylim(40, 110) +
  geom_density_2d_filled(alpha = 0.5)

# Boxplot ----------------------------------------------------------------------

ggplot(mpg, aes(class, cty)) +
  geom_boxplot(varwidth=T, fill="plum")





ggplot(mpg, aes(manufacturer, cty)) +
  geom_boxplot() + 
  geom_dotplot(binaxis='y', 
               stackdir='center', 
               dotsize = .5, 
               fill="red")


# Violin -----------------------------------------------------------------------

ggplot(mpg, aes(class, cty)) +
  geom_violin()




# Function----------------------------------------------------------------------
ggplot() + 
  xlim(-5, 5) + 
  geom_function(fun = function(x) 0.5*exp(-abs(x)))

# Map --------------------------------------------------------------------------
ids <- factor(c("1.1", "2.1", "1.2", "2.2", "1.3", "2.3"))

values <- data.frame(
  id = ids,
  value = c(3, 3.1, 3.1, 3.2, 3.15, 3.5)
)

positions <- data.frame(
  id = rep(ids, each = 4),
  x = c(2, 1, 1.1, 2.2, 1, 0, 0.3, 1.1, 2.2, 1.1, 1.2, 2.5, 1.1, 0.3,
        0.5, 1.2, 2.5, 1.2, 1.3, 2.7, 1.2, 0.5, 0.6, 1.3),
  y = c(-0.5, 0, 1, 0.5, 0, 0.5, 1.5, 1, 0.5, 1, 2.1, 1.7, 1, 1.5,
        2.2, 2.1, 1.7, 2.1, 3.2, 2.8, 2.1, 2.2, 3.3, 3.2)
)

ggplot(values, aes(fill = value)) +
  geom_map(aes(map_id = id), map = positions) +
  expand_limits(positions)
############################# EXTERNALS ########################################
# Marginal Histogram------------------------------------------------------------
library(ggExtra)
g <- ggplot(mpg, aes(cty, hwy)) + 
  geom_count() + 
  geom_smooth(method="lm", se=F)

ggMarginal(g, type = "histogram", fill="transparent")
ggMarginal(g, type = "boxplot", fill="transparent")


# Correlograma -----------------------------------------------------------------
library(ggcorrplot)
data(mtcars)
corr <- round(cor(mtcars), 1)

# Plot
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlogram of mtcars", 
           ggtheme=theme_bw)








































# Encircling -------------------------------------------------------------------
library(ggalt)
midwest_select <- midwest[midwest$poptotal > 350000 & 
                            midwest$poptotal <= 500000 & 
                            midwest$area > 0.01 & 
                            midwest$area < 0.1, ]

# Plot
ggplot(midwest, aes(x=area, y=poptotal)) + 
  geom_point(aes(col=state, size=popdensity)) +   # draw points
  geom_encircle(aes(x=area, y=poptotal), 
                data=midwest_select, 
                color="red", 
                size=2, 
                expand=0.08)

