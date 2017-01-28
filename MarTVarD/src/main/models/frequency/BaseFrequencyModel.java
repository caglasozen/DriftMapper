package main.models.frequency;

import main.report.ExperimentResult;
import main.models.Model;
import main.report.SingleExperimentResult;
import main.report.StructuredExperimentResult;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.math.BigInteger;
import java.util.*;

/**
 * Created by loongkuan on 28/11/2016.
 **/

public abstract class BaseFrequencyModel extends Model {

    protected BigInteger[] hashBases;

    public abstract int findFv(Instance instance, int[] attributesSubset);
    public abstract int findFy(int classIndex);
    public abstract int findFvy(Instance instance, int[] attributesSubset, int classIndex);
    protected abstract Set<BigInteger> getAllHashes(int[] attributeSubset);

    protected void setDataset(Instances dataset) {
        this.allInstances = new Instances(dataset, dataset.size());
        // Get bases for hash
        BigInteger base = BigInteger.ONE;
        this.hashBases = new BigInteger[dataset.numAttributes() - 1];
        for (int j : this.attributesAvailable) {
            this.hashBases[j] = base;
            base = base.multiply(BigInteger.valueOf(dataset.attribute(j).numValues()));
        }
        this.reset();
    }

    protected BigInteger instanceToPartialHash(Instance instance, int[] attributeSubset) {
        BigInteger hash = BigInteger.ZERO;
        // Iterate over attributes in strippedInstance
        for (int i : attributeSubset) {
            hash = hash.add(this.hashBases[i].multiply(BigInteger.valueOf((int)instance.value(i))));
        }
        return hash;
    }

    protected Instance partialHashToInstance(BigInteger partialHash, int[] activeAttributes) {
        Instance partialInstance = new DenseInstance(this.allInstances.numAttributes());
        partialInstance.setDataset(this.allInstances);

        for (int i = activeAttributes.length - 1; i >= 0; i--) {
            partialInstance.setValue(activeAttributes[i], partialHash.divide(this.hashBases[activeAttributes[i]]).intValue());
            partialHash = partialHash.mod(this.hashBases[activeAttributes[i]]);
        }
        return partialInstance;
    }

    protected boolean isPartialHashEqualHash(BigInteger partialHash, BigInteger hash, int[] attributeSubset) {
        BigInteger rem = hash.add(partialHash.negate());
        BigInteger quo;

        for (int i = this.allInstances.numAttributes() - 2; i >= 0; i--) {
            if (rem.compareTo(BigInteger.ZERO) == -1) return false;
            if (!ArrayUtils.contains(attributeSubset, i)) {
                quo = rem.divide(this.hashBases[i]);
                BigInteger numVals = BigInteger.valueOf(this.allInstances.attribute(i).numValues());
                if (quo.compareTo(numVals) >= 0) return false;
                rem = rem.remainder(this.hashBases[i]);
            }
        }
        return rem.compareTo(BigInteger.ZERO) == 0;
    }

    protected static BigInteger attributeSubsetToHash(int[] attributeSubset) {
        // TODO: this method skips over a lot of values since attribute subset is fixed length
        BigInteger hash = BigInteger.ZERO;
        for (int i : attributeSubset) {
            hash = hash.add(BigInteger.valueOf(2).pow(i));
        }
        return hash;
    }

    protected boolean validAttributeSubset(int[] attributeSubset) {
        boolean valid = true;
        for (int attributeIndex : attributeSubset) {
            valid = valid && ArrayUtils.contains(this.attributesAvailable, attributeIndex);
        }
        return valid && attributeSubset.length == this.attributeSubsetLength;
    }

    protected static ArrayList<BigInteger> sampleHashes(ArrayList<BigInteger> hashes, double sampleScale) {
        ArrayList<BigInteger> sampledHashes = new ArrayList<>();
        ArrayList<BigInteger> allHash = new ArrayList<>(hashes);
        Collections.shuffle(allHash);
        for (int i = 0; i < Math.min((int)(hashes.size() / sampleScale), hashes.size()); i++) {
            sampledHashes.add(allHash.get(i));
        }
        return sampledHashes;
    }

    protected ArrayList<BigInteger> mergeHashes(BaseFrequencyModel model, int[] attributeSubset) {
        Set<BigInteger> hashes = new HashSet<>();
        hashes.addAll(this.getAllHashes(attributeSubset));
        hashes.addAll(model.getAllHashes(attributeSubset));
        return new ArrayList<>(hashes);
    }

    protected ArrayList<BigInteger> intersectHashes(BaseFrequencyModel model, int[] attributeSubset) {
        Set<BigInteger> intersection = new HashSet<>();
        Set<BigInteger> modelHashes = model.getAllHashes(attributeSubset);

        for (BigInteger hash : this.getAllHashes(attributeSubset))  {
            if (modelHashes.contains(hash)) intersection.add(hash);
        }
        return new ArrayList<>(intersection);
    }

    @Override
    public double findPv(Instance instance, int[] attributesSubset) {
        return (double)findFv(instance, attributesSubset) / (double)this.size();
    }

    @Override
    public double findPy(int classIndex) {
        return (double)findFy(classIndex) / (double)this.size();
    }

    @Override
    public double findPvy(Instance instance, int[] attributesSubset, int classIndex) {
        return (double)findFvy(instance, attributesSubset, classIndex) / (double)this.size();
    }

    @Override
    public double findPvgy(Instance instance, int[] attributesSubset, int classIndex) {
        return (double)findFvy(instance, attributesSubset, classIndex) / (double)findFy(classIndex);
    }

    @Override
    public double findPygv(Instance instance, int[] attributesSubset, int classIndex) {
        return (double)findFvy(instance, attributesSubset, classIndex) / (double)findFv(instance, attributesSubset);
    }

    // TODO: Try and separate these into smaller functions
    @Override
    public ExperimentResult findCovariateDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        if (!validAttributeSubset(attributeSubset)) return null;
        BaseFrequencyModel modelAfter = (BaseFrequencyModel)modelToCompare;

        // Get the hash of all the seen instances and get a sample from them
        ArrayList<BigInteger> allHashes = mergeHashes(modelAfter, attributeSubset);
        allHashes = sampleHashes(allHashes, sampleScale);

        double[] p = new double[allHashes.size()];
        double[] q = new double[allHashes.size()];
        double[] separateDistance = new double[allHashes.size()];
        Instances instanceValues = new Instances(this.allInstances, allHashes.size());
        for (int i = 0; i < allHashes.size(); i++) {
            Instance instance = this.partialHashToInstance(allHashes.get(i), attributeSubset);
            p[i] = this.allInstances.size() == 0 ?
                    0 : (double)this.findFv(instance, attributeSubset) / (double)this.allInstances.size();
            q[i] = modelAfter.allInstances.size() == 0 ?
                    0 : (double)modelAfter.findFv(instance, attributeSubset) / (double)modelAfter.allInstances.size();
            separateDistance[i] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
            instanceValues.add(instance);
        }
        double finalDistance = this.distanceMetric.findDistance(p, q) * sampleScale;
        return new SingleExperimentResult(finalDistance, separateDistance, instanceValues);
    }

    @Override
    public ExperimentResult findJointDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        if (!validAttributeSubset(attributeSubset)) return null;
        BaseFrequencyModel modelAfter = (BaseFrequencyModel)modelToCompare;

        // Get the hash of all the seen instances and get a sample from them
        ArrayList<BigInteger> allHashes = mergeHashes(modelAfter, attributeSubset);

        int nClass =  this.allInstances.numClasses();
        double[] p = new double[allHashes.size() * nClass];
        double[] q = new double[allHashes.size() * nClass];
        double[] separateDistance = new double[allHashes.size() * nClass];
        Instances instanceValues = new Instances(this.allInstances, allHashes.size() * nClass);
        for (int i = 0; i < allHashes.size(); i++) {
            Instance instance = this.partialHashToInstance(allHashes.get(i), attributeSubset);
            for(int classIndex = 0; classIndex < nClass; classIndex++) {
                instance.setClassValue(classIndex);
                int currentIndex = i * nClass + classIndex;
                p[currentIndex] = this.allInstances.size() == 0 ?
                        0 : (double)this.findFvy(instance, attributeSubset, classIndex) / (double)this.allInstances.size();
                q[currentIndex] = modelAfter.allInstances.size() == 0 ?
                        0 : (double) modelAfter.findFvy(instance, attributeSubset, classIndex) / (double)modelAfter.allInstances.size();
                separateDistance[currentIndex] = this.distanceMetric.findDistance(new double[]{p[currentIndex]}, new double[]{q[currentIndex]});
                instanceValues.add(instance);
            }
        }
        double finalDistance = this.distanceMetric.findDistance(p, q) * sampleScale;
        return new SingleExperimentResult(finalDistance, separateDistance, instanceValues);
    }

    @Override
    public ExperimentResult findLikelihoodDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        if (!validAttributeSubset(attributeSubset)) return null;
        BaseFrequencyModel modelAfter = (BaseFrequencyModel)modelToCompare;

        // Get the hash of all the seen instances and get a sample from them
        ArrayList<BigInteger> allHashes = mergeHashes(modelAfter, attributeSubset);

        double[] p = new double[allHashes.size()];
        double[] q = new double[allHashes.size()];
        int nClass =  this.allInstances.numClasses();
        double[] separateDistance = new double[allHashes.size() * nClass];
        Instances instanceValues = new Instances(this.allInstances, allHashes.size() * nClass);
        double finalDistance = 0.0f;

        HashMap<BigInteger, ExperimentResult> separateExperiments = new HashMap<>();
        HashMap<BigInteger, Double> experimentProbability = new HashMap<>();

        for (int classIndex = 0; classIndex < this.allInstances.numClasses(); classIndex++) {
            double weight = (
                    ((double)this.findFy(classIndex) / (double)this.allInstances.size()) +
                            ((double)modelAfter.findFy(classIndex) / (double)modelAfter.allInstances.size())
            ) / 2;
            for (int i = 0; i < allHashes.size(); i++) {
                Instance instance = this.partialHashToInstance(allHashes.get(i), attributeSubset);
                instance.setClassValue(classIndex);
                p[i] = this.findFy(classIndex) == 0 ?
                        0 : (double) this.findFvy(instance, attributeSubset, classIndex) / (double) this.findFy(classIndex);
                q[i] = modelAfter.findFy(classIndex) == 0 ?
                        0 : (double) modelAfter.findFvy(instance, attributeSubset, classIndex) / (double) modelAfter.findFy(classIndex);
                separateDistance[classIndex * allHashes.size() + i] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
                instanceValues.add(instance);
            }
            double currentDistance = this.distanceMetric.findDistance(p, q) * sampleScale;
            experimentProbability.put(BigInteger.valueOf(classIndex), weight);
            separateExperiments.put(BigInteger.valueOf(classIndex),
                    new SingleExperimentResult(
                            currentDistance,
                            ArrayUtils.subarray(separateDistance,
                                    classIndex * allHashes.size(),
                                    (classIndex + 1) * allHashes.size()),
                            new Instances(instanceValues, classIndex * allHashes.size(), allHashes.size())));
            finalDistance += currentDistance * weight;
        }
        return new StructuredExperimentResult(finalDistance, separateDistance, instanceValues,
                separateExperiments, experimentProbability, new int[]{this.allInstances.classIndex()});
    }

    @Override
    public ExperimentResult findPosteriorDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        if (!validAttributeSubset(attributeSubset)) return null;
        BaseFrequencyModel modelAfter = (BaseFrequencyModel)modelToCompare;

        // Get the hash of all the seen instances and get a sample from them
        ArrayList<BigInteger> allHashes = mergeHashes(modelAfter, attributeSubset);

        double[] p = new double[this.allInstances.numClasses()];
        double[] q = new double[this.allInstances.numClasses()];
        int nClass = this.allInstances.numClasses();
        double[] separateDistance = new double[nClass * allHashes.size()];
        Instances instanceValues = new Instances(this.allInstances, allHashes.size() * nClass);
        double finalDistance = 0.0f;

        HashMap<BigInteger, ExperimentResult> separateExperiments = new HashMap<>();
        HashMap<BigInteger, Double> experimentProbability = new HashMap<>();

        for (int i = 0; i < allHashes.size(); i++) {
            Instance instance = this.partialHashToInstance(allHashes.get(i), attributeSubset);
            double weight = (((double)this.findFv(instance, attributeSubset)/(double)this.allInstances.size()) +
                    ((double)modelAfter.findFv(instance, attributeSubset)/(double)modelAfter.allInstances.size())) / 2;
            for (int classIndex = 0; classIndex < nClass; classIndex++) {
                instance.setClassValue(classIndex);
                p[classIndex] = this.findFv(instance, attributeSubset) == 0 ?
                        0 : (double) this.findFvy(instance, attributeSubset, classIndex) / (double) this.findFv(instance, attributeSubset);
                q[classIndex] = modelAfter.findFv(instance, attributeSubset) == 0 ?
                        0 : (double) modelAfter.findFvy(instance, attributeSubset, classIndex) / (double) modelAfter.findFv(instance, attributeSubset);
                separateDistance[i * nClass + classIndex] = this.distanceMetric.findDistance(new double[]{p[classIndex]}, new double[]{q[classIndex]});
                instanceValues.add(instance);
            }
            double currentDistance = this.distanceMetric.findDistance(p, q) * sampleScale;
            experimentProbability.put(allHashes.get(i), weight);
            separateExperiments.put(allHashes.get(i),
                    new SingleExperimentResult(
                            currentDistance,
                            ArrayUtils.subarray(separateDistance,
                                    i * nClass,
                                    (i + 1) * nClass),
                            new Instances(instanceValues, i * nClass, nClass)));
            finalDistance += currentDistance * weight;
        }
        return new StructuredExperimentResult(finalDistance, separateDistance, instanceValues,
                separateExperiments, experimentProbability, attributeSubset);
    }

    @Override
    public double peakCovariateDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        return this.findCovariateDistance(modelToCompare, attributeSubset, sampleScale).getDistance();
    }

    @Override
    public double peakJointDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        return this.findJointDistance(modelToCompare, attributeSubset, sampleScale).getDistance();
    }

    @Override
    public double peakLikelihoodDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        return this.findLikelihoodDistance(modelToCompare, attributeSubset, sampleScale).getDistance();
    }

    @Override
    public double peakPosteriorDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        return this.findPosteriorDistance(modelToCompare, attributeSubset, sampleScale).getDistance();
    }
}
