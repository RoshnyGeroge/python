Gather the coordinates as json from https://geojson.io/. eg. data/bus1.json

bus_event_producer: produces events based on bus route co-ordinates
run  topic_client to verify the result

app.py : flask application that acts as a client for topics.
run topic producer to generate events on topic