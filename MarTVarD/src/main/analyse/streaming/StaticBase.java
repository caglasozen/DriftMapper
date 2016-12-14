package main.analyse.streaming;

import main.models.Model;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by loongkuan on 2/12/2016.
 */
public class StaticBase extends StreamAnalysis{
    private int windowSize;
    private int currentIndex = 0;
    private Map<Integer, Double> driftTimeLine;

    public StaticBase(Instances baseInstances, Model referenceModel, int windowSize) {
        this.baseModel = referenceModel.copy();
        this.movingModel = referenceModel.copy();
        this.baseModel.addInstances(baseInstances);
        this.windowSize = windowSize;
        driftTimeLine = new HashMap<>();
        // TODO: make param
        this.resolution = windowSize / 5;
    }

    @Override
    public void reset() {
        this.baseModel.reset();
    }

    @Override
    public void addInstance(Instance instance) {
        this.movingModel.addInstance(instance);
        this.currentIndex += 1;
        if (this.movingModel.size() > this.windowSize) {
            for (int j = 0; j < this.resolution; j++) {
                this.movingModel.removeInstance(0);
            }
        }
        if (this.movingModel.size() % this.resolution == 0) {
            driftTimeLine.put(this.currentIndex,
                    this.baseModel.peakJointDistance(
                            this.movingModel,
                            this.baseModel.getAttributesAvailable(),
                            1.0));
        }
    }

    public void printDriftTimeLine() {
        for (int i = 0; i < this.currentIndex; i+=this.resolution) {
            System.out.println("Drift of " + this.driftTimeLine.get(i) + " at " + i);
        }
    }
}
