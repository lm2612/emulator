RegionLonsLats = {'US':(235.,290.,30.,50.),
        'Europe_Approx':(336.,30.,36.,66. ), # may need to treat differently because Europe lies over 0 degree longitude
        'Europe':(350.,40.,35.,70.),
        'NHML':(0.,360.,30.,60. ),
        'China':(80.,120.,20.,50. ),
        'East_Asia_Matt':(105.,145.,20.,45. ),
        'East_Asia':(100.,145.,20.,50. ),
        'India': (70.,90.,10.,30. ),
        'Asia': (60.,140.,10.,50.),
        'South_Asia': (70.,90.,10.,30. ), # Same box as India
        'Arabia': (30.,85.,0.,40.),
        'Global':(0.,360.,-90.,90.),
        'NH':(0.,360.,0.,90.),
        'SH':(0.,360.,-90.,90.),
        'Africa':(360.-20.,50.,-35.,35.),
        'South_America':(360.-80.,360.-35.,-50.,10.),
        'NHML_ext':(0.,360.,20.,70.),
        'Tropics':(0.,360.,-30.,30.),
        'SHML':(0.,360.,-60.,-30.),
        'SHML_ext':(0.,360.,-70.,-20.),
        'NHHL':(0.,360.,60.,90.),
        'SHHL':(0.,360.,-90.,60.),
        #'NP':(0.,360.,66.,90.),        # defined by arctic circle
        'SP':(0.,360.,-90.,-66.),
        'NorwSea':(10.,70.,70.,80.),    # Small blob in Norweigan sea appears influential
        'Sahel':(360.-17.,38.,9.,19.),
        'North_America':(360.-129.,360.-77.,32.,60.),
        'India/Bangladesh':(71.,94.,15.,28.), # Same as Shindel et al
        'Southwest_China/SE_Asia':(98.,110.,11.,30.),
        'Southwestern_US':(360-120.,360.-103.,32.,37),
        'Eastern_US':(360.-95., 360-77.,34. ,44. ),
        'Pacific_Northwest': (360.-129 ,360.-115,42 ,60.),
        'Arctic':(0.,360.,66.,90.),
        'Austrailia':(95.,155.,-40.,-10.),
        'Africa_mid':(360.-20.,50.,5.,10.),
        'Africa_desert':(360.-20.,30.,10.,35.),   #not included in emissions region, desert region
        'Africa_mid_east':(30.,50.,10.,35.),
        'Africa_south':(10.,50.,-35.,5.)
}

Regions = {'Arctic':(0.,360.,66.,90.),
        'NorthAmerica':(230.,300.,10.,65.),
        'NorthPacific':(145.  ,230.,10.,65.),
        'SouthPacific':(180., 360.-80., -50.,9.),
        'SouthAmerica':(360.-80.,360.-35.,-50.,9.),
        'Antarctic':(0.,360.,-90.,-66.),
        'SouthernOcean':(0.,360.,-66.,-51.),
        'SouthAtlantic':(360.-35.,10. ,-50.,9.),
        'NorthAtlantic':(300.,340.,10.,65.),
        'NorthernAfrica':(340.,50.,10.,34.), # would prefer this to start at 5 deg?
        'SouthernAfrica':(10.,50.,-50.,9.),
        'Europe':(340.,50.,35.,65.),
        'Russia':(50.,  100. ,35.,65.),
        'SouthAsia':(50.,100.,0.,34.),
        'IndianOcean':(50.,100.  ,-50. ,-1.),
        'Oceania':(100.,180. ,-50., 9.),
        'EastAsia':(100.,145.,10.,65. ),
        'Global':(-0.01,360.01,-90.05,90.05)}


RegionsFiner = {'Arctic0-90':(0.,90.,66.,90.),
        'Arctic90-180':(90.,180.,66.,90.),
        'Arctic180-270':(180.,270.,66.,90.),
        'Arctic180-270':(270.,360,66.,90.),
        'NorthAmericaE':(260.,300.,10.,65.),
        'NorthAmericaW':(230.,260.,10.,65.),
        'NorthPacificE':(145.  ,180.,10.,65.),
        'NorthPacificW':(180.  ,230.,10.,65.),
        'SouthPacificW':(180., 230., -50.,9.),
        'SouthPacificE':(230., 360.-80., -50.,9.),
        'SouthAmerica':(360.-80.,360.-35.,-50.,9.),
        'Antarctic0-90':(0.,90.,-90.,-66.),
        'SouthernOcean0-90':(0.,90.,-66.,-51.),
        'Antarctic90-180':(90.,180.,-90.,-66.),
        'SouthernOcean90-180':(90.,180.,-66.,-51.),
        'Antarctic180-270':(180.,270.,-90.,-66.),
        'SouthernOcean180-270':(180.,270.,-66.,-51.),
        'Antarctic270-360':(270.,360.,-90.,-66.),
        'SouthernOcean270-360':(270.,360.,-66.,-51.),
        'SouthAtlantic':(360.-35.,10. ,-50.,9.),
        'NorthAtlantic':(300.,340.,10.,65.),
        'NorthernAfrica':(340.,50.,10.,34.), # would prefer this to start at 5 deg?
        'SouthernAfrica':(10.,50.,-50.,9.),
        'Europe':(340.,50.,35.,65.),
        'Russia':(50.,  100. ,35.,65.),
        'SouthAsia':(50.,100.,0.,34.),
        'IndianOcean':(50.,100.  ,-50. ,-1.),
        'OceaniaE':(145.,180. ,-50., 9.),
        'OceaniaW':(100.,145. ,-50., 9.),
        'EastAsia':(100.,145.,10.,65. )}




WorldRegions = {
        'Global':(0.,360.,-90.,90.),
        'NH':(0.,360.,0.,90.),
        'SH':(0.,360.,-90.,90.),
        'NHML':(0.,360.,30.,60.),
        'SHML':(0.,360.,-60.,-30.),
        'NHHL':(0.,360.,60.,90.),
        'SHHL':(0.,360.,-90.,-60.),
        'Tropics':(0.,360.,-30.,30.)
}

RegionsList = Regions.keys()


