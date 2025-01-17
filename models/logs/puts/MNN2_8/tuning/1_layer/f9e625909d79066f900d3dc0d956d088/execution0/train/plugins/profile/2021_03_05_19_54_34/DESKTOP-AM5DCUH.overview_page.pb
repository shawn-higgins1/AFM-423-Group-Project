�	W#��2�7@W#��2�7@!W#��2�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-W#��2�7@>�#dx5@1�J��?A5�8EGr�?I-�i��� @*	������G@2U
Iterator::Model::ParallelMapV2���QI�?!��
��
>@)���QI�?1��
��
>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��@��ǈ?!"�k"�k9@)M�O��?1��7��75@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'�����?!2�z1�z6@)_�Q�{?1$I�$I�,@:Preprocessing2F
Iterator::Model��_�L�?!�F��F�E@)9��v��z?1��O��O+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!>�b>�b @)�q����o?1>�b>�b @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS�!�uq�?!�&�&L@)a��+ei?1W�V�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!a��`��@)����Mb`?1a��`��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!̧^̧^8@)��H�}M?1��@��@�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���X@QH���r�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	>�#dx5@>�#dx5@!>�#dx5@      ��!       "	�J��?�J��?!�J��?*      ��!       2	5�8EGr�?5�8EGr�?!5�8EGr�?:	-�i��� @-�i��� @!-�i��� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���X@yH���r�?�"5
sequential/dense/MatMulMatMul_zU�{�?!_zU�{�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�ބ�Í�?!v,���ֵ?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad1&h4��?!����B�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul1&h4��?!����W�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�ބ�Í�?!��PjU��?"!
Adam/PowPow�ބ�Í�?!kS!�{�?"#
Adam/addAddV2�ބ�Í�?!=��K��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�ބ�Í�?!�¼~��?"7
sequential/dense_1/MatMulMatMul�ބ�Í�?!�&�-70�?0"
Sum_7Sum��PjU�?!�n��V�?Q      Y@Y��/Ċ�0@a�	�N]�T@qj?gL.�T@yδ�!!�?"�
both�Your program is POTENTIALLY input-bound because 90.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�82.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 