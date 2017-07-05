package result.timeline;

import global.DriftMeasurement;

import java.util.Arrays;

/**
 * Created by LoongKuan on 2/07/2017.
 */
public class ConsolePrintListener implements TimelineListener{
    @Override
    public void updateMetaData(String[] attributeSubsets) {
        System.out.println(Arrays.toString(attributeSubsets));
    }

    @Override
    public DriftMeasurement getMeasurementType() {
        return DriftMeasurement.COVARIATE;
    }

    @Override
    public void returnDriftPointMagnitude(int driftPoint, double[] driftMagnitude) {
        System.out.println(driftPoint);
        System.out.println(Arrays.toString(driftMagnitude));
    }
}
