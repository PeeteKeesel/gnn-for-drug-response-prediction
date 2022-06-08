

- When I put `optimizer.zero_grad()` at the beginning of the loop the loss function goes all over the place. However when I put it like shown below the loss decreases iteratively.
```python
        optimizer.zero_grad()  # We need to zero the gradients, otherwise the next batch would also need to deal 
                               # with the previous batches gradients.
                               # This is because the calculated gradients accumulate by default.
        loss.backward()        # Compute the gradients.
        optimizer.step()       # Use the gradients and perform adjustment/update of the weights.
```