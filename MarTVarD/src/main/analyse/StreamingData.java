package main.analyse;

import main.models.Model;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by loongkuan on 29/11/2016.
 */
public class StreamingData {
    Instances currentWindow;
    Model baseModel;
    boolean baseStable;
    int maxWindowSize;
    HoeffdingAdaptiveTree tree = new HoeffdingAdaptiveTree();


    public void StreamDetection(int maxWindowSize) {
        this.maxWindowSize= maxWindowSize;

    }

    public void reset() {

    }

    public void addInstance(Instance instance) {
        if (baseModel == null) {
            currentWindow.add(instance);
        }

    }
}
