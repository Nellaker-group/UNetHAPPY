from shapely import geometry
import matplotlib.pyplot as plt
import math
from shapely.validation import make_valid, explain_validity

## eval run 24
## poly_id 899



poly = geometry.Polygon([[56445 ,13081], [ 56446 ,13080], [ 56446 ,13075], [ 56450 ,13073], [ 56453 ,13069], [ 56453 ,13065], [ 56454.666666666664 ,13063.333333333334], [ 56458 ,13062], [ 56462 ,13059], [ 56463 ,13054], [ 56467 ,13049], [ 56468 ,13044], [ 56469.142857142855 ,13042.57142857143], [ 56470 ,13042], [ 56470.46153846154 ,13040.923076923076], [ 56472 ,13039], [ 56477 ,13025], [ 56477 ,13016], [ 56478 ,13013], [ 56478 ,13012], [ 56478 ,13005], [ 56475 ,12997], [ 56474 ,12987], [ 56472 ,12984], [ 56470 ,12978], [ 56467 ,12972], [ 56459 ,12964], [ 56457.25 ,12963.25], [ 56457 ,12963], [ 56450 ,12960], [ 56446 ,12960], [ 56439 ,12957], [ 56438 ,12957], [ 56432 ,12957], [ 56429.333333333336 ,12953.666666666666], [ 56429 ,12953], [ 56428.69565217391 ,12952.869565217392], [ 56428 ,12952], [ 56422 ,12950], [ 56409 ,12950], [ 56401 ,12951], [ 56399 ,12952.5], [ 56398 ,12953], [ 56398.04347826087 ,12953.217391304348], [ 56397 ,12954], [ 56399 ,12958], [ 56398 ,12959], [ 56390 ,12959], [ 56389 ,12958], [ 56388 ,12955], [ 56385 ,12955], [ 56378 ,12957], [ 56371 ,12957], [ 56349 ,12962], [ 56343 ,12962], [ 56334 ,12966], [ 56332 ,12968], [ 56326 ,12971], [ 56316 ,12975], [ 56313 ,12977], [ 56307 ,12983], [ 56301 ,12985], [ 56295 ,12991], [ 56293 ,12992], [ 56289 ,12996], [ 56286 ,13000], [ 56275 ,13008], [ 56271 ,13015], [ 56268 ,13015], [ 56265 ,13018], [ 56264 ,13020], [ 56265 ,13025], [ 56264 ,13026], [ 56262 ,13025], [ 56261 ,13024], [ 56259 ,13024], [ 56256 ,13027], [ 56252 ,13034], [ 56244 ,13044], [ 56239 ,13053], [ 56236 ,13059], [ 56234 ,13065], [ 56233 ,13072], [ 56230 ,13079], [ 56231 ,13081], [ 56388 ,13081], [ 56443 ,13081], [ 56445 ,13081]])
poly2 = geometry.Polygon([[56415 ,13106], [ 56412 ,13106], [ 56407 ,13102], [ 56402 ,13102], [ 56392 ,13097], [ 56388 ,13097], [ 56387 ,13095], [ 56387 ,12960], [ 56388 ,12955], [ 56388 ,12954], [ 56388 ,12957], [ 56390 ,12959], [ 56393 ,12959], [ 56397 ,12960], [ 56398 ,12959], [ 56398 ,12957], [ 56396 ,12954], [ 56397 ,12953], [ 56430 ,12953], [ 56432 ,12954], [ 56431 ,12955], [ 56433 ,12955], [ 56433 ,12954], [ 56435 ,12954], [ 56437 ,12955], [ 56444 ,12959], [ 56450 ,12960], [ 56457 ,12962], [ 56465 ,12970], [ 56469 ,12976], [ 56470 ,12981], [ 56474 ,12987], [ 56474 ,12996], [ 56478 ,13004], [ 56478 ,13013], [ 56476 ,13016], [ 56476 ,13025], [ 56475 ,13028], [ 56471 ,13039], [ 56467 ,13045], [ 56466 ,13049], [ 56463 ,13053], [ 56461 ,13059], [ 56459 ,13061], [ 56454 ,13064], [ 56453 ,13069], [ 56446 ,13075], [ 56445 ,13079], [ 56442 ,13082], [ 56434 ,13090], [ 56431 ,13091], [ 56429 ,13093], [ 56427 ,13099], [ 56424 ,13100], [ 56422 ,13101], [ 56417 ,13105], [ 56415 ,13106]])

poly25 = geometry.Polygon([[56415 ,13106], [ 56412 ,13106], [ 56407 ,13102], [ 56402 ,13102], [ 56392 ,13097], [ 56388 ,13097], [ 56387 ,13095], [ 56474 ,12987], [ 56474 ,12996], [ 56478 ,13004], [ 56478 ,13013], [ 56476 ,13016], [ 56476 ,13025], [ 56475 ,13028], [ 56471 ,13039], [ 56467 ,13045], [ 56466 ,13049], [ 56463 ,13053], [ 56461 ,13059], [ 56459 ,13061], [ 56454 ,13064], [ 56453 ,13069], [ 56446 ,13075], [ 56445 ,13079], [ 56442 ,13082], [ 56434 ,13090], [ 56431 ,13091], [ 56429 ,13093], [ 56427 ,13099], [ 56424 ,13100], [ 56422 ,13101], [ 56417 ,13105], [ 56415 ,13106]])





x,y = poly.exterior.xy
plt.plot(x,y)
plt.savefig("badpolygonV2poly1.png")

x2,y2 = poly2.exterior.xy
plt.plot(x2,y2)
plt.savefig("badpolygonV2poly2.png")



x25,y25 = poly25.exterior.xy
plt.plot(x25,y25)
plt.savefig("badpolygonV2poly25.png")
plt.clf() 



polyS = poly.simplify(1.0, preserve_topology=False)
poly2S = poly2.simplify(1.0, preserve_topology=False)
poly25S = poly25.simplify(1.0, preserve_topology=False)


x,y = polyS.exterior.xy
plt.plot(x,y)
plt.savefig("badpolygonV2poly1simplified.png")

x2,y2 = poly2S.exterior.xy
plt.plot(x2,y2)
plt.savefig("badpolygonV2poly2simplified.png")

x25,y25 = poly25S.exterior.xy
plt.plot(x25,y25)
plt.savefig("badpolygonV2poly25simplified.png")
plt.clf()


x,y = poly.exterior.xy
plt.plot(x,y)
plt.savefig("badpolygonV2poly1.png")

x2,y2 = poly2.exterior.xy
plt.plot(x2,y2)
plt.savefig("badpolygonV2poly2.png")



x25,y25 = poly25.exterior.xy
plt.plot(x25,y25)
plt.savefig("badpolygonV2poly25.png")
plt.clf()



poly3 = geometry.Polygon([[56200 ,13100], [56300 ,13100], [56300 ,13000], [56200 ,13000]])
x3,y3 = poly3.exterior.xy
plt.plot(x3,y3)
plt.savefig("badpolygonV2poly3.png")

poly4 = geometry.Polygon([[56400 ,13100], [56500 ,13100], [56500 ,13000], [56400 ,13000]])
x4,y4 = poly4.exterior.xy
plt.plot(x4,y4)
plt.savefig("badpolygonV2poly4.png")

poly5 = geometry.Polygon([[56400 ,13000], [56450 ,13000], [56450 ,12950], [56400 ,12950]])
x5,y5 = poly5.exterior.xy
plt.plot(x5,y5)
plt.savefig("badpolygonV2poly5.png")



print("poly.is_valid")
print(poly.is_valid)


print("poly2.is_valid")
print(poly2.is_valid)


print("explain_validity(poly2)")
print(explain_validity(poly2))


print("poly.intersects(poly3)")
print(poly.intersects(poly3))


print("poly2.intersects(poly4)")
print(poly2.intersects(poly4))


print("poly2.intersects(poly5)")
print(poly2.intersects(poly5))


print("poly.intersects(poly25)")
print(poly.intersects(poly25))


print("polyS.intersects(poly2S)")
print(polyS.intersects(poly2S))



print("poly.intersects(poly2)")
print(poly.intersects(poly2))
