package result.timeline;

import global.DriftMeasurement;

/**
 * Created by LoongKuan on 21/06/2017.
 */
public interface TimelineListener {
    void updateMetaData(String[] attributeSubsets);
    void returnDriftPointMagnitude(int driftPoint, double[] driftMagnitude);
    DriftMeasurement getMeasurementType();
}
