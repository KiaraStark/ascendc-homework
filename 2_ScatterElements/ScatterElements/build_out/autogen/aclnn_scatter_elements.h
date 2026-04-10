
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SCATTER_ELEMENTS_H_
#define ACLNN_SCATTER_ELEMENTS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnScatterElementsGetWorkspaceSize
 * parameters :
 * varRef : required
 * indices : required
 * updates : required
 * axis : optional
 * reduceOptional : optional
 * varRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterElementsGetWorkspaceSize(
    aclTensor *varRef,
    const aclTensor *indices,
    const aclTensor *updates,
    int64_t axis,
    char *reduceOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnScatterElements
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterElements(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
